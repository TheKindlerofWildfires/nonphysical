use core::marker::PhantomData;

use super::hasher::FallHasher;

pub struct HashMap<K, V> {
    pub hash_builder: FallHasher,
    pub table: RawTable<(K, V)>,
}

#[derive(Clone)]
pub struct Keys<'a, K, V> {
    inner: Iter<'a, K, V>,
}
#[derive(Clone)]
pub struct Values<'a, K, V> {
    inner: Iter<'a, K, V>,
}
pub struct ValuesMut<'a, K, V> {
    inner: IterMut<'a, K, V>,
}
#[derive(Clone)]
pub struct Iter<'a, K, V> {
    inner: RawIter<(K, V)>,
    marker: PhantomData<(&'a K, &'a V)>,
}
pub struct IterMut<'a, K, V> {
    inner: RawIter<(K, V)>,
    marker: PhantomData<(&'a K, &'a mut V)>,
}
impl<K: Clone, V: Clone> Clone for HashMap<K, V> {
    fn clone(&self) -> Self {
        HashMap {
            hash_builder: self.hash_builder.clone(),
            table: self.table.clone(),
        }
    }
}

impl<K, V> HashMap<K, V> {
    pub fn new() -> Self {
        Self {
            hash_builder: FallHasher::default(),
            table: RawTable::new(),
        }
    }
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys { inner: self.iter() }
    }

    pub fn values(&self) -> Values<'_, K, V> {
        Values { inner: self.iter() }
    }

    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut {
            inner: self.iter_mut(),
        }
    }

    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            inner: self.table.iter(),
            marker: PhantomData,
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            inner: self.table.iter(),
            marker: PhantomData,
        }
    }
    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn entry(&mut self, key: K) -> Option<&V>{
        let hash = Self::make_hash(&key);
        self.table.find(hash,&key)
    }

    pub fn entry_ref(&mut self, key: K)-> Option<&mut V>{
        let hash = Self::make_hash(&key);
        self.table.find(hash,&key)
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let hash = Self::make_hash::<K, S>(&self.hash_builder, &k);
        let hasher = make_hasher::<_, V, S>(&self.hash_builder);
        match self
            .table
            .find_or_find_insert_slot(hash, equivalent_key(&k), hasher)
        {
            Ok(bucket) => Some(mem::replace(unsafe { &mut bucket.as_mut().1 }, v)),
            Err(slot) => {
                unsafe {
                    self.table.insert_in_slot(hash, slot, (k, v));
                }
                None
            }
        }
    }
    pub fn remove<Q>(&mut self, k: &Q) -> Option<V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.remove_entry(k) {
            Some((_, v)) => Some(v),
            None => None,
        }
    }

    pub fn remove_entry<Q>(&mut self, k: &Q) -> Option<(K, V)>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = make_hash::<Q, S>(&self.hash_builder, k);
        self.table.remove_entry(hash, equivalent_key(k))
    }
}
