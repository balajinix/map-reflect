import json
import os
import re
import time
import sqlite3
import shutil
import multiprocessing
import logging
from milvus_helper import MilvusHelper
#from queue import Queue
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

class Spider:
    def __init__(self, split='test'): 

        logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        if split == "dev":
            self.schema_file_path = './data/spider/tables.json'
            self.dataset_file_path = './data/spider/dev.json'
            self.gold_file_path = './data/spider/dev_gold.sql'
            self.databases_path = './data/spider/database/'
            self.gold_results_file = './data/spider/dev_gold_results.jsonl'
        elif split == "train":
            # TODO check this - shouldn't this be train?
            self.schema_file_path = './data/spider/tables.json'
            self.dataset_file_path = './data/spider/dev.json'
            self.gold_file_path = './data/spider/dev_gold.sql'
            self.databases_path = './data/spider/database/'
            self.gold_results_file = './data/spider/train_gold_results.jsonl'
        elif split == "test":
            self.schema_file_path = './data/spider/test_data/tables.json'
            self.dataset_file_path = './data/spider/test_data/dev.json'
            self.gold_file_path = './data/spider/test_data/dev_gold.sql'
            self.databases_path = './data/spider/test_database/'
            self.gold_results_file = './data/spider/test_data/test_gold_results.jsonl'

        self.collection_name = "spider_questions"
        self.schema_data = []
        self.data = []
        self.question_query_mapping = {}
        self.initialize()

        self.gold_results_map = self.load_gold_results(self.gold_results_file)
        self.save_interval = 100

    def get_gold_sql_queries(self):
        """
            get_gold_sql_queries
        """
        with open(self.gold_file_path, 'r') as f:
            return f.read().splitlines()

    def initialize(self):
        try:
            self.schema_data = self.load_schema()
            self.data = self.load_dataset()
            self.question_query_mapping = self.create_question_query_mapping()
        except Exception as e:
            print(f"Error initializing Spider: {e}")

    def load_schema(self):
        try:
            with open(self.schema_file_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading schema: {e}")
            return []

    def load_dataset(self):
        try:
            with open(self.dataset_file_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

    def create_question_query_mapping(self):
        mapping = {}
        for item in self.data:
            question = item.get('question')
            query = item.get('query')
            if question and query:
                mapping[question] = query
        return mapping

    def get_sql_for_question(self, question):
        return self.question_query_mapping.get(question)

    def get_database_names(self):
        try:
            return [db['db_id'] for db in self.schema_data if 'db_id' in db]
        except Exception as e:
            print(f"Error getting database names: {e}")
            return []

    def get_tables_columns(self, db_id):
        try:
            db_schema = next((item for item in self.schema_data if item['db_id'] == db_id), None)
            if db_schema:
                tables_columns = {}
                for idx, table_name in enumerate(db_schema['table_names']):
                    tables_columns[table_name] = [col[1] for col in db_schema['column_names'] if col[0] == idx]
                return tables_columns
        except Exception as e:
            print(f"Error getting tables columns for {db_id}: {e}")
        return {}

    def get_db_details(self, db_id):
        try:
            db_schema = next((db for db in self.schema_data if db['db_id'] == db_id), None)
            if db_schema:
                db_details = {
                    'db_id': db_id,
                    'tables': db_schema['table_names'],
                    'columns': {table: [] for table in db_schema['table_names']},
                    'column_names': db_schema['column_names'],
                    'column_types': db_schema['column_types'],
                    'primary_keys': db_schema['primary_keys'],
                    'foreign_keys': []
                }

                for col in db_schema['column_names']:
                    table_idx, col_name = col
                    if table_idx != -1:  # Exclude -1 index which refers to '*'
                        table_name = db_schema['table_names'][table_idx]
                        db_details['columns'][table_name].append(col_name)

                for fk in db_schema['foreign_keys']:
                    src_idx, tgt_idx = fk
                    src_table_idx, src_col_name = db_schema['column_names'][src_idx]
                    tgt_table_idx, tgt_col_name = db_schema['column_names'][tgt_idx]
                    src_table = db_schema['table_names'][src_table_idx]
                    tgt_table = db_schema['table_names'][tgt_table_idx]
                    db_details['foreign_keys'].append([(src_table, src_col_name), (tgt_table, tgt_col_name)])

                return db_details
        except Exception as e:
            print(f"Error getting database details for {db_id}: {e}")
        return {}


    def get_db_questions(self, db_id):
        try:
            return [{**item, 'name': item['db_id']} for item in self.data if 'db_id' in item and item['db_id'] == db_id]
        except Exception as e:
            print(f"Error getting questions for database {db_id}: {e}")
            return []

    # unique database ids
    def get_database_ids(self):
        try:
            return list(set(item['db_id'] for item in self.data))
        except Exception as e:
            print(f"Error getting database ids: {e}")
            return []

    def get_all_questions(self):
        """
        Returns a list of all questions in the dataset.
        """
        try:
            return [item['question'] for item in self.data]
        except Exception as e:
            print(f"Error getting all questions: {e}")
            return []

    def index_questions_in_milvus(self, milvus_collection_name):
        """
        Indexes questions from the Spider dataset in Milvus.
        """
        # for now not using the milvus_collection_name sent as input
        try:
            milvus_helper = MilvusHelper(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                         collection_name=self.collection_name,
                                         dim=384)

            questions = self.get_all_questions()
            embeddings = milvus_helper.model.encode(questions)

            milvus_helper.insert_data(list(range(len(questions))), embeddings)

            milvus_helper.disconnect()
        except Exception as e:
            print(f"Error indexing questions in Milvus: {e}")

    def search_related_questions(self, query, top_k):
        try:
            milvus_helper = MilvusHelper(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                         collection_name=self.collection_name,
                                         dim=384)
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = milvus_helper.search(query, search_params, top_k+2)

            all_questions = self.get_all_questions()

            related_questions = [all_questions[hit.id] for hits in results for hit in hits if all_questions[hit.id] != query]

            return related_questions[:top_k]
        except Exception as e:
            print(f"Error searching related questions in Milvus: {e}")
            return []

    def get_all_questions_and_queries(self):
        questions_and_queries = []
        for item in self.data:
            if 'question' in item and 'query' in item:
                questions_and_queries.append({
                    'question': item['question'],
                    'query': item['query'],
                    'db_id': item['db_id'] if 'db_id' in item else None
                })
        return questions_and_queries

    def generate_nary_facts(self):
        """
        generate_nary_facts
        """
        nary_facts = []
        database_names = self.get_database_names()

        try:
            for db_id in database_names:
                db_details = self.get_db_details(db_id)
                if not db_details:
                    continue

                # Handling tables and columns
                for table in db_details['tables']:
                    nary_facts.append({
                        "subject": db_id,
                        "predicate": "has_table",
                        "object": table,
                        "attributes": []
                    })
                    for column in db_details['columns'][table]:
                        nary_facts[-1]["attributes"].append({
                            "subject": table,
                            "predicate": "has_column",
                            "object": column
                        })

                try:
                    for fk_pair in db_details['foreign_keys']:
                        src_table, src_column = fk_pair[0]
                        tgt_table, tgt_column = fk_pair[1]

                        nary_facts.append({
                            "subject": src_table + "." + src_column,
                            "predicate": "foreign_key_to",
                            "object": tgt_table + "." + tgt_column
                        })
                except Exception as e:
                    print(f"Failed to generate facts related to foreign keys: {e}")
        except Exception as e:
            print(f"Failed to generate n-ary facts: {e}")

        return nary_facts

    def load_gold_results(self, gold_results_file):
        gold_results_map = {}
        if os.path.exists(gold_results_file):
            with open(gold_results_file, 'r') as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        for key, value in result.items():
                            gold_results_map[key] = value
                    except json.JSONDecodeError:
                        self.logger.error(f"Error decoding JSON from line: {line}")
        return gold_results_map

    def save_gold_results(self):
        with open(self.gold_results_file, 'w') as f:
            for key, value in self.gold_results_map.items():
                json_line = json.dumps({key: value})
                f.write(json_line + '\n')

    def save_results(self, results_file, results):
        temp_file = results_file + ".temp"
        with open(temp_file, 'w') as f:
            for result in results:
                try:
                    json_line = json.dumps(result)
                    f.write(json_line + '\n')
                except TypeError as e:
                    self.logger.error(f"Error serializing result: {result} - {e}")
        shutil.move(temp_file, results_file)

    def run_query(self, db_file, query):
        try:
            conn = sqlite3.connect(db_file, timeout=30)
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            conn.close()
            return result
        except Exception as e:
            return e


    @staticmethod
    def process_sql(sql, db_id, is_gold=False):
        """
        process_sql
        """
        if sql is None:
            return sql

        try:
            if is_gold:
                sql = sql.lower().replace(f"\t{db_id}", "").strip()
                sql = sql.replace("  ", " ")
                sql = sql.replace("( ", "(")
                sql = sql.replace(" )", ")")
                sql = sql.replace("'", "\"")
                sql = sql.replace(";", "")
                return sql

            quoted_text = re.findall(r"(['\"])(.*?)(\1)", sql)

            placeholders = [f"quote_placeholder_{i}" for i in range(len(quoted_text))]
            sql_with_placeholders = sql

            for i, quote in enumerate(quoted_text):
                # quote[0] is the opening quote, quote[1] is the content, quote[2] is the closing quote
                full_quote = f"{quote[0]}{quote[1]}{quote[2]}"
                sql_with_placeholders = sql_with_placeholders.replace(full_quote, placeholders[i], 1)

            # Convert everything outside of quotes to lowercase
            sql_with_placeholders = sql_with_placeholders.lower()

            # Reinsert the quoted text back into the query by replacing placeholders
            for i, quote in enumerate(quoted_text):
                sql_with_placeholders = sql_with_placeholders.replace(placeholders[i], f"{quote[0]}{quote[1]}{quote[2]}")

            sql = sql_with_placeholders

            select_index = sql.lower().find("select")
            if select_index != -1:
                sql = sql[select_index:]

            # Remove content after triple backticks if present
            triple_backticks_index = sql.find("```")
            if triple_backticks_index != -1:
                sql = sql[:triple_backticks_index]

            sql = sql.replace(",", " , ").strip()
            sql = sql.split(";", 1)[0]
            sql = sql.split("\n\n\n", 1)[0]
            sql = sql.replace("\n", " ")
            sql = sql.replace("  ", " ")
            sql = sql.replace("'", "\"")
            sql = sql.replace(";", "")

            # Remove specific substrings
            answer_index = sql.lower().find(" answer")
            if answer_index != -1:
                sql = sql[:answer_index]

            prompt_index = sql.lower().find(" please")
            if prompt_index != -1:
                sql = sql[:prompt_index]

            prompt_index = sql.lower().find(" generate a correct sql query")
            if prompt_index != -1:
                sql = sql[:prompt_index]

            question_index = sql.lower().find(" question:")
            if question_index != -1:
                sql = sql[:question_index]

            explanation_index = sql.lower().find(" explanation:")
            if explanation_index != -1:
                sql = sql[:explanation_index]

            note_index = sql.lower().find(" note:")
            if note_index != -1:
                sql = sql[:note_index]

            if sql.startswith('"') and sql.endswith('"'):
                sql = sql[1:-1]

            # Remove <|endoftext|> if it occurs at the end of the SQL
            if sql.endswith('<|endoftext|>'):
                sql = sql[:-len('<|endoftext|>')].strip()

            sql = sql.strip()
        except Exception as e:
            print(str(e))
        return sql

    def process_query(self, result, db_file, timeout_sec):
        db_id = result["db_id"]
        gold_sql = result["gold_sql"]
        processed_gold_sql = self.process_sql(result["gold_sql"], db_id, is_gold=True)
        generated_sql = result["generated_sql"]
        processed_generated_sql = self.process_sql(result["generated_sql"], db_id)

        result["processed_generated_sql"] = processed_generated_sql

        # Check if the gold SQL results are already cached
        #gold_results = self.gold_results_map.get(gold_sql)
        gold_results = None

        # Execute gold query if not cached
        if gold_results is None:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.run_query, db_file, processed_gold_sql)
                    gold_results = future.result(timeout=timeout_sec)
                gold_error = isinstance(gold_results, Exception) or gold_results is None
                if not gold_error:
                    self.gold_results_map[processed_gold_sql] = gold_results
            except TimeoutError:
                self.logger.error(f"Gold SQL query timed out for db_id: {db_id}")
                result["execution_error"] = "Gold SQL query timed out"
                gold_error = True
            except Exception as e:
                self.logger.error(f"Gold SQL error for db_id: {db_id}: {e}")
                result["execution_error"] = "Gold SQL error: " + str(e)
                gold_error = True
        else:
            gold_error = False

        result["valid_query"] = not gold_error

        try:
            if len(processed_generated_sql) < 10:
                result["execution_error"] = "Invalid sql"
                result["execution_match"] = 0
                result["execution_bool"] = 0
            else:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.run_query, db_file, processed_generated_sql)
                    generated_results = future.result(timeout=timeout_sec)
                result["execution_bool"] = 1 if generated_results is not None and not isinstance(generated_results, Exception) else 0
                if not gold_error and generated_results is not None and gold_results == generated_results:
                    result["execution_match"] = 1
                    result['gold_results'] = str(gold_results)
                    result['generated_results'] = str(generated_results)
                else:
                    result['gold_results'] = str(gold_results)
                    result['generated_results'] = str(generated_results)
                    result["execution_match"] = 0
        except TimeoutError:
            self.logger.error(f"Generated SQL query timed out for db_id: {db_id}")
            result["execution_error"] = "Generated SQL query timed out"
            result["execution_match"] = 0
            result["execution_bool"] = 0
        except Exception as e:
            self.logger.error(f"Generated SQL error for db_id: {db_id}: {e}")
            result["execution_error"] = str(e)
            result["execution_match"] = 0
            result["execution_bool"] = 0

        return result

    def calculate_execution_accuracy(self, results_file, timeout_sec=30):
        """
        """
        results = []
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        results.append(result)
                    except json.JSONDecodeError:
                        self.logger.error(f"Error decoding JSON from line: {line}")

        print("#####")
        print(len(results))
        print("#####")
        valid_queries = 0
        execution_matches = 0
        output_results = []

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = []
            for idx, result in enumerate(tqdm(results, desc="Submitting queries")):
                db_id = result["db_id"]
                db_file = os.path.join(self.databases_path, f"{db_id}/{db_id}.sqlite")
                if not os.path.exists(db_file):
                    self.logger.warning(f"Database file {db_file} not found. Skipping.")
                    continue
                futures.append(executor.submit(self.process_query, result, db_file, timeout_sec))

            for idx, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Collecting execution results")):
                try:
                    result = future.result(timeout=timeout_sec)
                    if result["valid_query"]:
                        valid_queries += 1
                    if result.get("execution_match") == 1:
                        execution_matches += 1
                    output_results.append(result)
                    #self.logger.info(f"Output results length: {len(output_results)}")

                    if (len(output_results) % self.save_interval) == 0:
                        # Periodically save the gold results map
                        self.save_gold_results()
                        # Periodically save the current results
                        self.save_results(results_file, output_results)
                        #self.logger.info("Periodically saving results")

                except TimeoutError:
                    self.logger.error("A query exceeded the timeout and was terminated.")
                except Exception as e:
                    self.logger.error(f"An error occurred: {e}")

        self.logger.info("Execution of SQL queries complete")
        execution_accuracy = execution_matches / valid_queries if valid_queries > 0 else 0
        print(f"spider: execution_matches: {execution_matches}")
        print(f"spider: valid_queries: {valid_queries}")

        self.save_results(results_file, output_results)
        self.save_gold_results()
        self.logger.info("Final results saved")

        return execution_accuracy

    def classify_query(self, query):
        if 'JOIN' in query.upper():
            return 'Join'
        elif re.search(r"SELECT COUNT\(\*\) FROM", query, re.IGNORECASE):
            return 'Aggregation'
        elif re.search(r"SELECT .* WHERE .* IN \(SELECT .* FROM", query, re.IGNORECASE):
            return 'Nested'
        elif 'GROUP BY' in query.upper():
            return 'GroupBy'
        elif 'HAVING' in query.upper():
            return 'Having'
        elif 'ORDER BY' in query.upper():
            return 'Order'
        elif 'LIMIT' in query.upper():
            return 'Limit'
        elif 'DISTINCT' in query.upper():
            return 'Distinct'
        elif 'WHERE' in query.upper():
            return 'Condition'
        else:
            return 'Simple'

    def annotate_dataset(self):
        annotated_data = []
        for item in self.data:
            query = item.get('query', '')
            annotation = self.classify_query(query)
            annotated_item = item.copy()
            annotated_item['output_type'] = annotation
            annotated_data.append(annotated_item)
        return annotated_data

    def print_stats_for_splits(self, annotated_files):
        """
        Print output type statistics and total queries for each dataset split.
        """
        for split_name, file_path in annotated_files.items():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                stats = self.calculate_output_type_stats(data)
    
                total_queries = len(data)
                print(f"\nStats for {split_name.upper()}:")
                print(f"Total Queries: {total_queries}")
                for output_type, count in stats.items():
                    print(f"{output_type}: {count}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    def calculate_output_type_stats(self, data):
        """
        Calculate statistics of output types in the dataset.
        """
        stats = {}
        for item in data:
            output_type = item.get('output_type', 'unknown')
            if output_type not in stats:
                stats[output_type] = 1
            else:
                stats[output_type] += 1
        return stats
 
    def annotate_and_save_splits(self, splits, output_dir):
        """
            annotate
        """
        annotated_files = {}
        for split_name in splits:
            spider_instance = Spider(split_name)
    
            annotated_data = spider_instance.annotate_dataset()
    
            annotated_file_path = f"{output_dir}/annotated_{split_name}.json"
            with open(annotated_file_path, 'w') as f:
                json.dump(annotated_data, f, indent=4)
    
            annotated_files[split_name] = annotated_file_path
        return annotated_files
    
    def add_ids_to_queries(self, split_name, output_dir):
        spider_instance = Spider(split=split_name)
    
        input_file = spider_instance.gold_file_path
        output_file = f"{output_dir}/spider_{split_name}_queries_with_ids.txt"
    
        with open(input_file, 'r') as input_f, open(output_file, 'w') as output_f:
            query_id = 0
    
            for line in input_f:
                line = line.strip()
                output_f.write(f"{query_id}\t{line}\n")
                query_id += 1
    
        print(f"Queries with IDs for split '{split_name}' written to '{output_file}'")
    
    
    def add_ids_to_queries_dedup(self, split_name, output_dir):
        spider_instance = Spider(split=split_name)
    
        input_file = spider_instance.gold_file_path
        output_file = f"{output_dir}/spider_{split_name}_queries_with_ids.txt"
    
        seen = set()
        query_id = 0
    
        with open(input_file, 'r') as input_f, open(output_file, 'w') as output_f:
            for line in input_f:
                stripped_line = line.strip()
                if stripped_line not in seen:
                    seen.add(stripped_line)
                    output_f.write(f"{query_id}\t{stripped_line}\n")
                    query_id += 1
    
        print(f"Queries with IDs for split '{split_name}' written to '{output_file}' without duplicates")

if __name__ == "__main__":
    splits = ["dev", "train", "test"]
    #annotated_files = annotate_and_save_splits(splits, output_dir='./data/spider/')
    #print_stats_for_splits(annotated_files)

    output_dir = './data/spider/'
    for split_name in splits:
        spider = Spider(split_name)
        spider.add_ids_to_queries_dedup(split_name, output_dir)
