from libcpp.unordered_set cimport unordered_set
from libcpp cimport bool
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
import cython

cdef extern from "LSH.h":
    cdef cppclass LSH:
        LSH(int, int, int) except +
        void remove(int*, int) except +
        void insert(int*, int) except +
        void insert_multi(int*, int) except +
        unordered_set[int] query(int*) except +
        unordered_set[int] query_multi(int*, int) except +
        void query_multi_mask(int*, float*, int, int) except +
        vector[unordered_set[int]] query_multiset(int*, int) except +
        # void query_multi_mask_L(int*, float*,float*, int, int) except +
        void clear() except +
        vector[int] print_stats()

cdef class pyLSH:
    cdef LSH* c_lsh

    def __cinit__(self, int K, int L, int THREADS):
        self.c_lsh = new LSH(K, L, THREADS)

    def __dealloc__(self):
        del self.c_lsh

    @cython.boundscheck(False)
    def remove(self, np.ndarray[int, ndim=1, mode="c"] fp, int item_id):
        self.c_lsh.remove(&fp[0], item_id)

    @cython.boundscheck(False)
    def insert(self, np.ndarray[int, ndim=1, mode="c"] fp, int item_id):
        self.c_lsh.insert(&fp[0], item_id)

    @cython.boundscheck(False)
    def insert_multi(self, np.ndarray[int, ndim=2, mode="c"] fp, int N):
        self.c_lsh.insert_multi(&fp[0, 0], N)

    @cython.boundscheck(False)
    def query(self, np.ndarray[int, ndim=1, mode="c"] fp):
        return self.c_lsh.query(&fp[0])

    @cython.boundscheck(False)
    def query_multi(self, np.ndarray[int, ndim=2, mode="c"] fp, int N):
        return self.c_lsh.query_multi(&fp[0, 0], N)

    @cython.boundscheck(False)
    def query_multi_mask(self, np.ndarray[int, ndim=2, mode="c"] fp, np.ndarray[float, ndim=2, mode="c"] mask, int M, int N):
        return self.c_lsh.query_multi_mask(&fp[0, 0], &mask[0,0], M, N)

#    @cython.boundscheck(False)
#    def query_multi_mask_L(self, np.ndarray[int, ndim=2, mode="c"] fp, np.ndarray[float, ndim=2, mode="c"] mask, np.ndarray[float, ndim=3, mode="c"] mask_L, int M, int N):
#        return self.c_lsh.query_multi_mask_L(&fp[0, 0], &mask[0,0], &mask_L[0,0,0], M, N)

    @cython.boundscheck(False)
    def accidental_match(self, np.ndarray[long, ndim=1, mode="c"] labels, set samples, int N):
        for idx in range(N): 
            if labels[idx] in samples:
                samples.remove(labels[idx])

    @cython.boundscheck(False)
    def multi_label(self, np.ndarray[long, ndim=2, mode="c"] labels, set samples):
        M = labels.shape[0]
        K = labels.shape[1]
        label2idx = dict()
        label_list = list()
        batch_prob = list()

        # remove accidental hits from samples
        # create label list
        # create label to index dictionary
        for idx in range(M): 
            count = 0
            for jdx in range(K): 
                l = labels[idx, jdx]
                if l == -1:
                    break
                elif l in samples:
                    samples.remove(l)
                count += 1
                if l not in label2idx:
                    label2idx[l] = len(label_list)
                    label_list.append(l)
            batch_prob.append(1.0 / count)

        sample_list = label_list + list(samples)

        # create probability distribution
        result = np.zeros([M, len(sample_list)], dtype=np.float32)
        for idx in range(M): 
            for jdx in range(K): 
                l = labels[idx, jdx]
                if l == -1:
                    break
                else:
                    #result[idx, label2idx[l]] = batch_prob[idx]
                    result[idx, label2idx[l]] = batch_prob[idx]
        return sample_list, result

    @cython.boundscheck(False)
    def multi_label_nonunion(self, np.ndarray[long, ndim=2, mode="c"] labels, np.ndarray[long, ndim=2, mode="c"] samples):
        M = labels.shape[0]
        K = labels.shape[1]
        num_class = samples.shape[1]


        # remove accidental hits from samples
        # create label list
        # create label to index dictionary
        label_count =np.zeros(M)
        for idx in range(M): 
            for jdx in range(K): 
                l = labels[idx, jdx]
                if l == -1:
                    label_count[idx] = jdx
                    break
                if(jdx == K-1):
                    label_count[idx] = K
                samples[idx][l] = 0
        

        
        max_padding = max(np.sum(samples,axis=1) + label_count).astype("int")
        sample_list = np.zeros((M, max_padding)) + num_class

        label_count = label_count.astype("int")
        result = np.zeros([M, max_padding], dtype=np.float32)

        for idx in range(M):
            content = np.concatenate( [ labels[idx][labels[idx]>=0], np.squeeze(np.argwhere( samples[idx] >0 ))])
            sample_list[idx,0: len(content)]  = content
            result[idx, 0:label_count[idx]] = 1/label_count[idx]
        
        return sample_list, result
        
    @cython.boundscheck(False)
    def query_matrix(self, np.ndarray[int, ndim=2, mode="c"] fp, np.ndarray[int, ndim=2, mode="c"] labels, int N, int total_size):
        multiset = self.c_lsh.query_multiset(&fp[0, 0], N)

        cdef total_count = 0
        cdef max_size = 0
        cdef np.ndarray local_label = np.zeros( labels.shape[1])
        cdef int temp_label = 0
        for idx in range(len(multiset)):
            local_label = labels[idx]

            for lidx in range(len(local_label)):
                temp_label = local_label[lidx]
                if( temp_label != total_size):

                    multiset[idx].erase(temp_label)

            total_count += len(multiset[idx])
            max_size = max(max_size, len(multiset[idx]))

        np_lsh = np.zeros([N, max_size], dtype=np.int64)
        np_lsh.fill(total_size)
        for bdx, item in enumerate(multiset):
            for ldx, index in enumerate(item):
                np_lsh[bdx, ldx] = index
        return np_lsh, total_count

    def print_stats(self):
        return self.c_lsh.print_stats()

    def clear(self):
        self.c_lsh.clear()
