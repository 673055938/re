The source code used in paper "Towards Empty Answers in SPARQL: Approximating Querying with RDF Embedding"

Dateset: DBpedia english version. Training the hole dataset will cost you too much time. You can use SPARQL to extract a subset, and then feed it to the embedding model.

Twenty failing queries are reported in "failing queries.xlsx";


Corrections to a error in my paper:

The following error was introduced during the editing/proofing stages. Negative sampling Equation (5) 

<img src="https://github.com/wangmengsd/re/blob/master/e.png" width="450"/>

should be replaced by:

<img src="https://github.com/wangmengsd/re/blob/master/r.png" width="450"/>


Other graph-embedding based KG query paper:

*Embedding Logical Queries on Knowledge Graphs. William L. Hamilton, Marinka Zitnik, Payal Bajaj, Dan Jurafsky, Jure Leskovec. In Proceedings of NIPS. Dec. 2018. 

*TrQuery: An Embedding-based Framework for Recommanding SPARQL Queries. 	Lijing Zhang, Xiaowang Zhang, Zhiyong Feng. In Proceedings of ICTAI. Nov. 2018.
