#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>

class LSH
{
	private:
		// Members
		int counter;
		int rnd;
		const int K;
		const int L;
		const int THREADS;
		std::vector<std::unordered_map<int, std::vector<int>>> tables;

		// Functions
		void add(const int, const int, const int);
		void add_multi(const int*, const int, const int);

		void retrieve(bool*, const int, const int);
		void retrieve(std::unordered_set<int>&, const int, const int);
		void retrieve_mask(const int*, float*, const int, const int, const int);
		// void retrieve_mask(const int*, float*, float*, const int, const int, const int);

	public:
		LSH(int, int, int);
        void remove(const int*, int);
		void insert(const int*, const int);
		void insert_multi(const int*, const int);
		std::unordered_set<int> query(const int*);
		std::unordered_set<int> query_multi(const int*, const int);
		void query_multi_mask(const int*, float*, const int, const int);
		void query_multi_mask_L(const int*, float*, float*,const int, const int);
		std::vector<std::unordered_set<int>> query_multiset(const int* fp, const int N);
		void clear();
		std::vector<int> print_stats();
};
