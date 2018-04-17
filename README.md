The source code used in paper "Towards Empty Answers in SPARQL: Approximating Querying with RDF Embedding"

Dateset: DBpedia english version. Training the hole dataset will cost you too much time. You can use SPARQL to extract a subset, and then feed it to the embedding model.

Training Parameters are here (you can also copy it in trainConfig.json file direclty)

      dataset": "resource",

      "n":"50",

      "rate":"0.001",

      "epochNum":"1000",

      "batchNum":"1000",

      "threadNum":"16",

      "EntityContextPath":"EntityContext.json",

      "entity2VecPath":"",

      "relation2VecPath":"",

      "outDir":"out"


Twenty failing queries are reported in "failing queries.xlsx";
