use std::collections::VecDeque;

pub(crate) struct ScoreCache<T> {
    pub(crate) capacity: usize,
    pub(crate) entries: VecDeque<(String, T)>,
}

impl<T: Clone> ScoreCache<T> {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: VecDeque::with_capacity(capacity.min(128)),
        }
    }

    pub(crate) fn get(&mut self, key: &str) -> Option<T> {
        let index = self
            .entries
            .iter()
            .position(|(entry_key, _)| entry_key == key)?;
        let (key, value) = self.entries.remove(index)?;
        let cloned = value.clone();
        self.entries.push_front((key, value));
        Some(cloned)
    }

    pub(crate) fn insert(&mut self, key: String, value: T) {
        if self.capacity == 0 {
            return;
        }
        if let Some(index) = self
            .entries
            .iter()
            .position(|(entry_key, _)| entry_key == &key)
        {
            let _ = self.entries.remove(index);
        }
        self.entries.push_front((key, value));
        while self.entries.len() > self.capacity {
            let _ = self.entries.pop_back();
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}
