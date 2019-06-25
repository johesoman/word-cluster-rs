#![allow(dead_code)]
#![feature(duration_float)]

use std::mem;
use std::fs::File;
use std::path::Path;
use std::error::Error;
use std::iter::FromIterator;
use std::io::{BufRead, BufReader, BufWriter, Write};

mod edit_dist;

extern crate rand;
use rand::prelude::*;

extern crate serde;
use serde::{Serialize, Deserialize};

extern crate rayon;
use rayon::prelude::*;

// ++++++++++
// + Galaxy +
// ++++++++++
struct SliceGalaxy<'a> {
    medoids: Vec<&'a [char]>,
    clusters: Vec<Vec<&'a [char]>>
}

impl<'a> SliceGalaxy<'a> {
    fn to_string_galaxy(&self) -> StringGalaxy {
        let medoids : Vec<String> = self.medoids.iter()
            .map(|x| String::from_iter(x.iter()))
            .collect();

        let clusters : Vec<Vec<String>> = self.clusters.iter()
            .map(|xs|
                    xs.iter()
                    .map(|x| String::from_iter(x.iter()))
                    .collect()
                )
            .collect();

        StringGalaxy {
            medoids: medoids,
            clusters: clusters,
        }
    }

    fn merge_many(gxs: Vec<SliceGalaxy<'a>>) -> SliceGalaxy<'a> {
        let num_cls = gxs.iter()
            .map(|gx| gx.clusters.len())
            .sum();

        let mut gx_out = SliceGalaxy {
            medoids: Vec::with_capacity(num_cls),
            clusters: Vec::with_capacity(num_cls),
        };

        for gx in gxs {
            gx_out.medoids.extend(gx.medoids);
            gx_out.clusters.extend(gx.clusters);
        }

        gx_out
    }
}

struct VecGalaxy {
    medoids: Vec<Vec<char>>,
    clusters: Vec<Vec<Vec<char>>>,
}

#[derive(Serialize, Deserialize)]
struct StringGalaxy {
    medoids: Vec<String>,
    clusters: Vec<Vec<String>>,
}

impl StringGalaxy {
    fn read_from_file<P: AsRef<Path>>(path: P)
         -> Result<StringGalaxy, Box<Error>>
    {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let gx = serde_json::from_reader(reader)?;
        Ok(gx)
    }

    fn write_to_file<P: AsRef<Path>>(&self, path: P)
        -> Result<(), Box<Error>>
    {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);

        serde_json::to_writer_pretty(writer, &self)?;
        Ok(())
    }

    fn data(self) -> (Vec<Vec<char>>, Vec<Vec<char>>) {
        let medoids = self.medoids.iter()
            .map(|x| x.chars().collect())
            .collect();

        let clusters = self.clusters.iter()
            .flat_map(|xs|
                    xs.iter().map(|x| x.chars().collect())
                )
            .collect();

        (medoids, clusters)
    }
}

// ++++++++++++++
// + data setup +
// ++++++++++++++

fn read_dict<S>(reader : BufReader<S>) -> Vec<Vec<char>>
    where
    S : std::io::Read
{
    let mut dict = Vec::<Vec<char>>::with_capacity(370_099);

    for line in reader.lines() {
        dict.push(line.unwrap().chars().collect())
    }

    dict
}

fn read_dict_from_file<P : AsRef<Path>>(path: P) -> Vec<Vec<char>> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);

    read_dict(reader)
}

fn setup_from_galaxy<P : AsRef<Path>>(path: P)
    -> (Vec<Vec<char>>, Vec<Vec<char>>)
{
    let gx = StringGalaxy::read_from_file(path)
        .map_err(|x| {println!("{}", x); x})
        .unwrap();

    let (mut medoids, mut words) = gx.data();
    medoids.sort_unstable();
    words.sort_unstable();
    (medoids, words)
}

fn setup_from_dict(num_medoids: usize, num_words: usize, dict: &[Vec<char>])
    -> (Vec<Vec<char>>, Vec<Vec<char>>) {

    let mut rng = rand::thread_rng();

    let dict : Vec<&[char]> = dict
        .choose_multiple(&mut rng, num_words)
        .map(|x| x.as_slice())
        .collect();

    let mut medoids : Vec<Vec<char>> = dict.iter()
        .take(num_medoids)
        .map(|x| x.to_vec())
        .collect();

    medoids.sort_unstable();

    let mut words : Vec<Vec<char>> = dict.iter()
        .take(num_words)
        .map(|x| x.to_vec())
        .collect();

    words.sort_unstable();

    (medoids, words)
}

// +++++++
// + pam +
// +++++++

fn pam<'a>(medoids: Vec<&'a [char]>, dict: &[&'a [char]]) -> SliceGalaxy<'a>
{
    let mut medoids = medoids;
    let mut medoids2 : Vec<&'a [char]> = Vec::with_capacity(medoids.len());
    let mut clusters : Vec<Vec<&'a [char]>> = vec![Vec::new(); medoids.len()];
    let mut belongs_to : Vec<usize> = Vec::with_capacity(dict.len());

    let mut it = 0..;

    loop {
        println!("pam @ i: {}", it.next().unwrap());

        // assign all words to closest medoid
        let it = dict.par_iter()
            .map(|word| {
                let (min_idx, _) =
                    edit_dist::index_of_nearest_word(word, &medoids[..]);

                min_idx
            });

        belongs_to.par_extend(it);

        clusters.par_iter_mut().enumerate().for_each(|(i, cl)| {
            belongs_to.iter().enumerate().filter(|(_, k)| i == **k).
                for_each(|(j, _)| { cl.push(&dict[j][..]); })
        });

        // for all cluster, the most central word becomes the new medoid
        let it = clusters.par_iter()
            .map(|cl| {
                let (min_word, _) = edit_dist::medoid_by_total_dist(cl);

                min_word
            });

        medoids2.par_extend(it);

        // let count : usize = medoids.iter().zip(medoids2.iter())
            // .map(|(x, y)| if x == y { 0 } else { 1 })
            // .sum();

        // println!("count: {}", count);

        // if new medoids are the same as old medoids, we are done
        if medoids == medoids2 { break; }

        // reset data for next iteration
        mem::swap(&mut medoids, &mut medoids2);
        medoids2.clear();
        belongs_to.clear();
        clusters.par_iter_mut().for_each(|cl| { cl.clear(); });
    }

    SliceGalaxy {
        medoids: medoids,
        clusters: clusters,
    }
}

fn minipam<'a>(
    i: usize,
    dummy: &[char],
    other: &[char],
    belongs_to: &mut Vec<usize>,
    cluster: &mut Vec<&'a [char]>,
    dict: &[&'a [char]])
{
    let mut medoid = other;
    let mut medoid2: &[char];
    let mut medoids: [&[char]; 2] = [&[]; 2];

    let mut it = 0..;

    loop {
        // println!("api @ i: {}, minipam @ j: {}", i, it.next().unwrap());

        // reset
        belongs_to.clear();
        cluster.clear();

        medoids[0] = dummy;
        medoids[1] = medoid;

        // assign all words to closest medoid
        let it = dict.par_iter()
            .map(|word| {
                let (min_idx, _) =
                    edit_dist::index_of_nearest_word(word, &medoids[..]);
                min_idx
            });

        belongs_to.par_extend(it);

        // build cluster
        let it = belongs_to.par_iter().enumerate()
            .filter(|(_, j)| **j == 1)
            .map(|(i, _)| dict[i]);

        cluster.par_extend(it);

        // find new medoid
        let (min_word, _) = edit_dist::medoid_by_total_dist_par(&cluster[..]);

        medoid2 = min_word;

        if medoid == medoid2 { break; }

        // reset
        mem::swap(&mut medoid, &mut medoid2);
    }
}

fn api<'a>(min_num_medoids: usize, dict: &[&'a [char]]) -> Vec<&'a [char]> {
    let mut dict: Vec<&[char]> = dict.iter().map(|x| &x[..]).collect();
    let mut dict2: Vec<&[char]> = Vec::with_capacity(dict.len());
    let mut belongs_to: Vec<usize> = Vec::with_capacity(dict.len());
    let mut cluster: Vec<&[char]> = Vec::new();
    let mut medoids: Vec<&'a [char]> = Vec::new();

    println!("api @ finding dummy...");

    let (min_word, _) = edit_dist::medoid_by_total_dist_par(&dict[..]);
    let dummy = min_word;

    let exp_cl_size = dict.len() / min_num_medoids;
    let mut diff_counter = 0;

    let mut it = 0..;

    loop {
        let i = it.next().unwrap();
        println!("api @ i: {}, dict.len(): {}", i, dict.len());

        let (min_idx, _) = edit_dist::index_of_farthest_word(dummy, &dict[..]);
        let other = dict[min_idx];

        minipam(i, dummy, other, &mut belongs_to, &mut cluster, &dict[..]);

        // remove other's cluster
        let it = belongs_to.par_iter().enumerate()
            .filter(|(_, j)| **j == 0)
            .map(|(i, _)| dict[i]);

        dict2.par_extend(it);

        if medoids.len() < min_num_medoids {
            medoids.push(other);
        } else if exp_cl_size <= diff_counter {
            diff_counter = 0;
            medoids.push(other);
        } else {
            diff_counter += dict.len() - dict2.len();
        }

        // if there are not enough left to cluster, we are done
        if dict2.len() <= exp_cl_size || dict2.len() == 1 {
            medoids.push(dummy);
            break;
        }

        // reset
        mem::swap(&mut dict, &mut dict2);
        dict2.clear();
    }

    println!("number of medoids found: {}", medoids.len());

    medoids
}

fn api_pam<'a>(min_num_medoids: usize, dict: &[&'a [char]]) -> SliceGalaxy<'a> {
    let medoids = api(min_num_medoids, dict);
    pam(medoids, dict)
}

fn cluster<'a>(min_num_medoids: usize, dict: &Vec<&'a [char]>)
    -> SliceGalaxy<'a>
{
    // TODO: code for splitting clusters that are too large
    let gx = api_pam(min_num_medoids, dict);
    gx
}

fn cluster_many<'a>(min_num_medoids: usize, dicts: Vec<Vec<&'a [char]>>)
    -> SliceGalaxy<'a>
{
    let tot_num_words : usize = dicts.iter()
        .map(|dict| dict.len())
        .sum();

    let mut it = b'a'..;

    let gxs = dicts.iter()
        .map(|dict| {
            let rate = dict.len() as f64 / tot_num_words as f64;
            let min_num_medoids = (min_num_medoids as f64 * rate) as usize;

            println!("at letter: {}", it.next().unwrap() as char);
            cluster(min_num_medoids, dict)
        })
        .collect();

   SliceGalaxy::merge_many(gxs)
}

struct Dictionary (Vec<Vec<char>>);

impl Dictionary {
    fn new(dict: Vec<Vec<char>>) -> Self {
        let mut dict = dict;
        dict.sort_unstable();
        Dictionary(dict)
    }

    fn slices(&self) -> Vec<&[char]> {
        self.0.iter().map(|x| &x[..]).collect()
    }

    fn slices_grouped_by_first(&self) -> Vec<Vec<&[char]>> {
        let dict = &self.0;

        if dict.is_empty() { return Vec::new(); }

        let mut slices: Vec<Vec<&[char]>> = Vec::with_capacity(26);

        let mut it = 0..;
        let mut i = 0;
        let mut prev: char = '\0';

        for word in dict.iter() {
            if word[0] != prev {
                prev = word[0];
                i = it.next().unwrap();
                slices.push(Vec::new());
            }
            slices[i].push(&word[..]);
        }

        slices
    }

    fn cluster<'a>(&'a self, min_num_medoids: usize) -> SliceGalaxy<'a> {
        cluster_many(min_num_medoids, self.slices_grouped_by_first())
    }
}

fn run_bench(
    num_threads: Option<usize>,
    min_num_medoids: usize,
    dict: &Dictionary,
    )
{
    num_threads.iter().for_each(|x| {
        rayon::ThreadPoolBuilder::new().num_threads(*x).build_global().unwrap()
    });

    println!("threads: {}", rayon::current_num_threads());

    let t = std::time::Instant::now();

    let gx = dict.cluster(min_num_medoids);

    let t = t.elapsed().as_float_secs();

    println!("time: {:.2}s", t);

    let gx = gx.to_string_galaxy();

    println!("number of clusters: {}", gx.clusters.len());
    println!("{:?}", gx.write_to_file("gx0.txt"));
}

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    let reader = std::io::stdin();
    let mut reader = BufReader::new(reader.lock());
    let writer = std::io::stdout();
    let mut writer = BufWriter::new(writer.lock());

    let dict = read_dict_from_file("dictionary.txt");
    // let (medoids, dict) = setup_from_galaxy("from_sam/gx0.txt");
    // let (medoids, dict) = setup_from_dict(0, 200000, &dict[..]);

    let dict = Dictionary::new(dict);

    let min_num_medoids = 15000;
    // let medoids = medoids.iter().map(|x| &x[..]).collect();
    run_bench(None, min_num_medoids, &dict);

    writer.write_all(b"\n").unwrap();
    writer.write_all(b"Goodbye!").unwrap();

    writer.flush().unwrap();
}
