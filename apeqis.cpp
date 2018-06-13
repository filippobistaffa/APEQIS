#include "apeqis.h"

#define PTR(VAR) .VAR = &VAR

struct mcnet_data {

	vector<vector<agent>> *pos, *neg;
	vector<value> *values;
};

__attribute__((always_inline)) inline
void createedge(edge *g, agent v1, agent v2, edge e) {

	#ifdef DOT
	printf("\t%u -- %u [label = \"e_%u,%u\"];\n", v1, v2, v1, v2);
	#endif
	g[v1 * _N + v2] = g[v2 * _N + v1] = e;
}

void parse_mcnet(const char *filename, edge *g, vector<vector<agent> > &pos, vector<vector<agent> > &neg, vector<value> &values) {

	ifstream f(filename);
	string line;
	#ifndef FULL_GRAPH
	edge ne = 0;
	#endif

	for (id i = 0; i < RULES; ++i) {

		getline(f, line);
		values[i] = stof(line);
		vector<agent> p, n;

		while (getline(f, line)) {

			if (line.compare("") == 0) {
				break;
			} else {
				p.push_back(stoi(line));
			}
		}

		while (getline(f, line)) {

			if (line.compare("") == 0) {
				break;
			} else {
				n.push_back(stoi(line));
			}
		}

		sort(p.begin(), p.end());
		sort(n.begin(), n.end());
		pos[i] = p;
		neg[i] = n;

		#ifndef FULL_GRAPH
		for (vector<agent>::const_iterator it1 = p.begin(); it1 != p.end(); ++it1) {
			for (vector<agent>::const_iterator it2 = it1 + 1; it2 != p.end(); ++it2) {
				if (!g[*it1 * _N + *it2]) {
					createedge(g, *it1, *it2, _N + ne);
					ne++;
				}
			}
		}
		#endif
	}

	f.close();
}

template <typename iterator1, typename iterator2>
__attribute__((always_inline)) inline
bool empty_intersection(iterator1 start1, iterator1 end1, iterator2 start2, iterator2 end2) {

	iterator1 i = start1;
	iterator2 j = start2;

	while (i != end1 && j != end2) {
		if (*i == *j) {
			return false;
		} else if (*i < *j) {
			++i;
		} else {
			++j;
		}
  	}
	return true;
}

value mcnet(agent *c, agent nl, void *data) {

	mcnet_data *mcnd = (mcnet_data *)data;
	value ret = 0;

	#ifdef APE_DEBUG
	printbuf(c + 1, c[0]);
	#endif

	for (id i = 0; i < RULES; ++i) {
		if (includes(c + 1, c + c[0] + 1, (*mcnd->pos)[i].begin(), (*mcnd->pos)[i].end()) &&
		    empty_intersection(c + 1, c + c[0] + 1, (*mcnd->neg)[i].begin(), (*mcnd->neg)[i].end())) {
			#ifdef APE_DEBUG
			cout << "Rule " << i << " applies" << endl;
			#endif
			ret += (*mcnd->values)[i];
		}
	}

	#ifdef APE_DEBUG
	cout << "Value = " << ret << endl;
	#endif

	return ret;
}

int main(int argc, char *argv[]) {

	// positive literals
	vector<vector<agent>> pos(RULES, vector<agent>());
	// negative literals
	vector<vector<agent>> neg(RULES, vector<agent>());
	// values
	vector<value> values(RULES);
	// mcnet graph
	edge *g = (edge *)calloc(_N * _N, sizeof(edge));

	#ifdef DOT
	cout << "graph G {" << endl;
	for (id i = 0; i < _N; ++i) {
		cout << "\t" << i << ";" << endl;
	} 
	#endif

	// parse mcnet
	parse_mcnet(argv[1], g, pos, neg, values);
	mcnet_data mcnd = { PTR(pos), PTR(neg), PTR(values) };

	#ifdef DOT
	cout << "}" << endl << endl;
	#endif

	#ifdef APE_DEBUG
	for (id i = 0; i < RULES; ++i) {
		cout << "Rule " << i << endl << "Value = " << values[i] << endl;
		printvec(pos[i], "Positive literals");
		printvec(neg[i], "Negative literals");
		cout << endl;
	}
	#endif

	#ifdef FULL_GRAPH
	edge ne = 0;

	for (id i = 0; i < _N; ++i) {
		for (id j = i + 1; j < _N; ++j) {
			g[i * _N + j] = g[j * _N + i] = _N + ne;
			ne++;
		}
	}
	#endif

	value *w = apeqis(g, mcnet, &mcnd, nullptr, _N, _N, argv[2], argv[3]);

	free(g);
	if (w) {
		free(w);
	}

	return 0;
}
