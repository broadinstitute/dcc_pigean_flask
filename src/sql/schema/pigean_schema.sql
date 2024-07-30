
-- schema for the pigean application

-- phenotype tables
drop table nmf_ontology_gene;
CREATE TABLE IF NOT EXISTS nmf_ontology_gene (
    id INTEGER PRIMARY KEY, 
    gene_synonym TEXT, 
    gene_name TEXT, 
    query_ontology_id TEXT
);



