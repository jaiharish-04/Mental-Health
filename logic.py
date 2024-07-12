import mysql.connector as mc


class process:
    def __init__(self):
        self.mycon = mc.connect(host="127.0.0.1", port=3306, user="root", passwd="Harish04!", database="miniproj")
        self.mycursor = self.mycon.cursor(buffered=True)

    def mysql_config(self):
        cmd = "create table users(name varchar(30), age int(10), gender char(6), email varchar(30), phone_num int(12), proff char(30),  password varchar(30))"
        self.mycursor.execute(cmd)

    def register(self, details):  # the param is a list
        name, age, gender, phone_num, proff, email, password = details

        print(self.mycon)
        cmd = "insert into users(name, age , gender , email, phone_num , proff, password) values(%s,%s,%s,%s,%s,%s,%s)"
        data_tup = (name, age, gender, email, phone_num, proff, password)
        self.mycursor.execute(cmd, data_tup)
        self.mycon.commit()

        return True

    def auth(self, creds):
        email, password = creds

        cmd = "select password from users where email=(%s)"
        dt = (email, )
        self.mycursor.execute(cmd, dt)

        lst = []
        for i in self.mycursor:
            lst.append(i)

        print(lst)
        db_pass = ""

        for index, tup in enumerate(lst):
            db_pass = tup[0]

        return db_pass == password

    def ret_name_from_email(self, email):
        cmd = "select name from users where email=(%s)"
        dt = (email, )
        self.mycursor.execute(cmd, dt)

        lst = []
        for i in self.mycursor:
            lst.append(i)

        print(lst)
        name = ""

        for index, tup in enumerate(lst):
            name = tup[0]
        return name