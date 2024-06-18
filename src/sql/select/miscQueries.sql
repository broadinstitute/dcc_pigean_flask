


select node_code from comb_node_ontology where node_type_id = 2 order by node_code;

-- mysql> desc comb_node_ontology;
-- +-------------------+---------------+------+-----+-------------------+-----------------------------------------------+
-- | Field             | Type          | Null | Key | Default           | Extra                                         |
-- +-------------------+---------------+------+-----+-------------------+-----------------------------------------------+
-- | id                | int           | NO   | PRI | NULL              | auto_increment                                |
-- | node_code         | varchar(500)  | NO   | MUL | NULL              |                                               |
-- | node_type_id      | int           | NO   | MUL | NULL              |                                               |
-- | ontology_id       | varchar(50)   | YES  | MUL | NULL              |                                               |
-- | ontology_type_id  | varchar(50)   | YES  |     | NULL              |                                               |
-- | node_name         | varchar(1000) | YES  |     | NULL              |                                               |
-- | last_updated      | datetime      | YES  |     | CURRENT_TIMESTAMP | DEFAULT_GENERATED on update CURRENT_TIMESTAMP |
-- | added_by_study_id | int           | YES  |     | NULL              |                                               |
-- +-------------------+---------------+------+-----+-------------------+-----------------------------------------------+
-- 8 rows in set (0.00 sec)
