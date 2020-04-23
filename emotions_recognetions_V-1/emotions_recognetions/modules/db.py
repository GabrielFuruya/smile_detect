# #databa

import psycopg2


class Database :

    def __init__ (self):

        self.connection = psycopg2.connect(

                        host = '172.31.1.210',
                        user = 'roit_service',
                        password = 'bWvn9fx5zC49PM',
                        dbname = 'artificial_intelligence',
                        port = '48864',
                    )
        return

    def insert_db(self,filename,date,faces,box,predict):
        self.cursor = self.connection.cursor()

        self.postgres_insert_query = """ INSERT INTO roit_ai_dev.detect_emotions (predict, file_name, "date", box, faces) VALUES (%s,%s,%s,%s,%s)"""
        self.record_to_insert = (predict,filename,date,box,faces)
        self.cursor.execute(self.postgres_insert_query, self.record_to_insert)

        self.postgreSQL_select_Query = "select * from roit_ai_dev.detect_emotions"

        self.cursor.execute(self.postgreSQL_select_Query)

        self.connection.commit()

        self.mobile_records = self.cursor.fetchall() 

    
        self.connection.close()
        return 
        