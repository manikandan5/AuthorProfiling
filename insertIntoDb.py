from pymongo import MongoClient
import os

# Insert into Database
def insert():
    client = MongoClient('localhost:27017', serverSelectionTimeoutMS=1000)

    db = client['user-details']
    collection = db['status']

    os.chdir(os.path.join(os.getcwd(), 'processed-data'))

    folders = os.listdir()

    s_no = 1

    print("Loading data into MongoDB")
    for folder in folders:
        temp = folder.split('-')
        if len(temp) > 2:
            for status in os.listdir(folder):
                file = open(folder + '/' + status, 'r')
                status_content = file.read()
                file.close()
                status = status.split('.')[0]
                result = db.collection.insert_one(
                    {
                        "SNo": s_no,
                        "StatusID": status,
                        "Status": status_content,
                        "Age": temp[1] + '-' + temp[2],
                        "Sex": temp[0]
                    })
                if s_no%10000 == 0:
                    print(".",end="")
                s_no += 1
    print()
    client.close()