use std::mem;
use std::cmp::min;

extern crate rayon;
use rayon::prelude::*;

// +++++++++++
// + helpers +
// +++++++++++

#[inline(always)]
fn min_max(x: usize, y: usize) -> (usize, usize) {
    if x < y { (x, y) } else { (y, x) }
}

// +++++++++++++
// + edit_dist +
// +++++++++++++

const WORD_LEN : usize = 40;

fn edit_dist<T : Eq>(
    tab: &mut [[u8; WORD_LEN]; WORD_LEN],
    prev: &[T],
    curr: &[T],
    word: &[T],
) -> usize {

    let k = prev
        .iter()
        .zip(curr.iter())
        .take_while(|(x, y)| x == y)
        .count();

    for (i, x) in curr.iter().enumerate().skip(k) {
        for (j, y) in word.iter().enumerate() {
            let d1 = tab[i][j] + (x != y) as u8;
            let d2 = tab[i][j + 1] + 1;
            let d3 = tab[i + 1][j] + 1;

            tab[i + 1][j + 1] = min(d1, min(d2, d3));
        }
    }

    tab[curr.len()][word.len()] as usize
}

// +++++++++++++++
// + Accumulator +
// +++++++++++++++

trait Accumulator<'a, T : Eq> {
   fn should_do(&self, curr: &[T], word: &[T]) -> bool;
   fn should_stop(&self) -> bool;
   fn accumulate(&mut self, idx: usize, curr: &'a [T], dist: usize);
}

fn run_accumulator<'a, S, T : Eq + 'a, I> (
    accumulator : &mut S,
    word: &[T],
    dict: I,
)
    where
    S : Accumulator<'a, T>,
    I : Iterator<Item = &'a [T]>
{
    let mut tab = [[0; WORD_LEN]; WORD_LEN];

    for i in 0..WORD_LEN { tab[i][0] = i as u8; }
    for j in 0..WORD_LEN { tab[0][j] = j as u8; }

    let mut prev: &[T] = &[];

    for (i, curr) in dict.enumerate() {
        if accumulator.should_stop() { break; }

        if accumulator.should_do(curr, word) {
            let dist = edit_dist(&mut tab, prev, curr, word);
            accumulator.accumulate(i, curr, dist);
            prev = curr;
        }
    }
}

// ++++++++++++++
// + spellcheck +
// ++++++++++++++

const SUGGESTIONS_LEN : usize = 32;

struct SuggestionList<'a, T : Eq> {
    suggestions: [&'a [T]; SUGGESTIONS_LEN],
    min_dist: usize,
    len: usize,
}

impl<'a, T : Eq> SuggestionList<'a, T> {
    pub fn new() -> SuggestionList<'a, T> {
        SuggestionList {
            suggestions: [&[]; SUGGESTIONS_LEN],
            min_dist: std::usize::MAX,
            len: 0,
        }
    }
}

impl<'a, T : Eq> Accumulator<'a, T> for SuggestionList<'a, T> {
   #[inline(always)]
    fn should_do(&self, curr : &[T], word: &[T]) -> bool {
        let (n, m) = min_max(curr.len(), word.len());
        m - n <= self.min_dist
    }

   #[inline(always)]
    fn should_stop(&self) -> bool { self.min_dist == 0 }

   #[inline(always)]
    fn accumulate(&mut self, _idx: usize, curr : &'a [T], dist : usize) {
        if dist < self.min_dist {
            self.min_dist = dist;
            self.suggestions[0] = curr;
            self.len = 1;
        } else if dist == self.min_dist {
            self.suggestions[self.len] = curr;
            self.len += 1;
            self.len &= SUGGESTIONS_LEN - 1;
        }
    }
}

pub fn spellcheck<'a, T : Eq>(
    word: &[T],
    dict: &Vec<&'a [T]>
) -> Vec<&'a [T]>
{
    let mut sl = SuggestionList::new();
    let it = dict.iter().map(|x| &x[..]);
    run_accumulator(&mut sl, word, it);

    sl.suggestions.iter()
        .take(sl.len)
        .map(|x| &x[..])
        .collect()
}

// +++++++++++++++++++++++++
// + index of farthes word +
// +++++++++++++++++++++++++

struct IndexOfFarthestWord (usize, usize);

impl IndexOfFarthestWord {
    fn new() -> IndexOfFarthestWord {
        IndexOfFarthestWord(0, 0)
    }
}

impl<'a, T : Eq> Accumulator<'a, T> for IndexOfFarthestWord {
    #[inline(always)]
    fn should_do(&self, _curr: &[T], _word: &[T]) -> bool { true }

    #[inline(always)]
    fn should_stop(&self) -> bool { false }

    #[inline(always)]
    fn accumulate(&mut self, idx: usize, _curr: &[T], dist: usize) {
        if self.1 < dist {
            mem::replace(self, IndexOfFarthestWord(idx, dist));
        }
    }
}

pub fn index_of_farthest_word<T : Eq>(word: &[T], dict: &[&[T]])
    -> (usize, usize)
{
    let mut acc = IndexOfFarthestWord::new();
    let it = dict.iter().map(|x| &x[..]);
    run_accumulator(&mut acc, word, it);
    (acc.0, acc.1)
}

// +++++++++++++++++++++++++
// + index of nearest word +
// +++++++++++++++++++++++++

struct IndexOfNearestWord (usize, usize);

impl IndexOfNearestWord {
    fn new() -> IndexOfNearestWord {
        IndexOfNearestWord(0, std::usize::MAX)
    }
}

impl<'a, T : Eq> Accumulator<'a, T> for IndexOfNearestWord {
    #[inline(always)]
    fn should_do(&self, curr: &[T], word: &[T]) -> bool {
        let (n, m) = min_max(curr.len(), word.len());
        m - n <= self.1
    }

    #[inline(always)]
    fn should_stop(&self) -> bool { self.1 == 0 }

    #[inline(always)]
    fn accumulate(&mut self, idx: usize, _curr: &[T], dist: usize) {
        if dist < self.1 {
            mem::replace(self, IndexOfNearestWord(idx, dist));
        }
    }
}

pub fn index_of_nearest_word<T : Eq>(word: &[T], dict: &[&[T]])
    -> (usize, usize)
{
    let mut acc = IndexOfNearestWord::new();
    let it = dict.iter().map(|x| &x[..]);
    run_accumulator(&mut acc, word, it);
    (acc.0, acc.1)
}

// ++++++++++++++
// + total dist +
// ++++++++++++++

struct TotalDist (usize, usize);

impl TotalDist {
    fn new(limit: usize) -> TotalDist { TotalDist(0, limit) }
}

impl<'a, T : Eq> Accumulator<'a, T> for TotalDist {
    #[inline(always)]
    fn should_do(&self, _curr: &[T], _word: &[T]) -> bool { true }

    #[inline(always)]
    fn should_stop(&self) -> bool { self.1 <= self.0 }

    #[inline(always)]
    fn accumulate(&mut self, _idx: usize, _curr: &[T], dist: usize) {
        mem::replace(self, TotalDist(self.0 + dist, self.1));
    }
}

// +++++++++++++++++++++
// + medoid predicates +
// +++++++++++++++++++++

pub fn medoid_by_total_dist<'a, T : Eq>(cluster: &[&'a [T]])
    -> (&'a [T], usize)
{
    let mut min_word: &[T] = &[];
    let mut min_dist: usize = std::usize::MAX;

    for word in cluster.iter() {
        let mut td = TotalDist::new(min_dist);
        let it = cluster.iter().map(|x| &x[..]);
        run_accumulator(&mut td, word, it);

        let dist = td.0;

        if dist < min_dist {
            min_word = word;
            min_dist = dist;
        }
    }

    (min_word, min_dist)
}

pub fn medoid_by_total_dist_par<'a, T : Eq + Sync>(cluster: &[&'a [T]])
    -> (&'a [T], usize)
{
    let init_word: &[T] = &[];
    let init_dist: usize = std::usize::MAX;

    let (min_word, min_dist) = cluster.par_iter()
        .map(|word| {

            let mut td = TotalDist::new(init_dist);
            let it = cluster.iter().map(|x| &x[..]);
            run_accumulator(&mut td, word, it);

            let dist = td.0;

            (word, dist)
        })
        .reduce(|| (&init_word, init_dist), |(s, x), (t, y)| {
            if x < y { (s, x) } else { (t, y) }
        });

    (min_word, min_dist)
}
