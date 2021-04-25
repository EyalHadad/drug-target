import csv
import os
import psycopg2
import pandas as pd
class DataConnector(object):

    def __init__(self, user, password):
        """
        :param user: SQL server user name.
        :param password: user's password.

        :return: DataWriter object

        Example::
            dw = DataWriter(user="userName", password="Password")
        """
        self.schema = None
        self.user = user
        self.password = password
        self.table = None
        self.columns = None
        self.connection = None
        self.cursor = None

    def connect(self):
        """
        connect to the sql server with the username and password that insert in initialization.
        The operation requires a BGU VPN connection.
        :return: None

        Example::
            dw.connect()
        """
        self.connection = psycopg2.connect(
            host="132.72.64.83",
            database="postgres",
            user=self.user,
            password=self.password,
        )
        self.cursor = self.connection.cursor()

    def insert_values(self, c_reader):
        postgres_insert_query = f'INSERT INTO {self.table} ({",".join(self.columns)}) VALUES(%s,%s,%s)'
        print(postgres_insert_query)
        for line in c_reader:
            record_to_insert = line[0].split("\t")
            self.cursor.execute(postgres_insert_query, record_to_insert)
            self.connection.commit()

    def create_table_command(self):
        columns_with_type = [x + " Text" for x in self.columns]
        columns_with_type[0] = columns_with_type[0].replace("Text", "Text PRIMARY KEY")
        command = f'create table {self.table} ({",".join(columns_with_type)});'
        print(command)
        return command

    def get_table(self, schema,table, dst_file_path,headers):
        print("downloading",schema,table, "data")
        get_table_command = f'select * from "{schema}"."{table}"'
        self.cursor.execute(get_table_command)
        mobile_records = pd.DataFrame(self.cursor.fetchall())
        mobile_records = mobile_records.apply(lambda x: x.astype(str).str.lower())
        print("creating ", dst_file_path, "file")
        mobile_records.to_csv(dst_file_path,header=headers, index=False)
        print(dst_file_path, "created successfully!\n")

    def upload_csv(self, schema, csv_file_path):
        """
        upload_csv method upload the content of .csv file to new table in the relevant schema.

        :param schema: existing schema name in postgres database.
        :param csv_file_path: path to the source csv file.

        :return: None

        Example::
            dw.upload_csv(schema="schema_name", f_path=r'csv_file_path')
        """
        self.schema = schema
        with open(csv_file_path, newline='') as csv_file:
            basename = os.path.basename(csv_file_path)
            if '.' in basename:
                basename = basename.split('.')[0]

            self.table = """%s.%s""" % (self.schema, basename)
            c_reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
            self.columns = next(c_reader)[0].split("\t")
            create_table = self.create_table_command()
            self.cursor.execute(create_table)
            self.connection.commit()
            self.insert_values(c_reader)

    def disconnect(self):
        """
        disconnect the sql server.

        :return: None

        Example::
            dw.disconnect()
        """
        self.connection.close()
