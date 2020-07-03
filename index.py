# coding=utf8
import MySQL

def index():
    mysql_20 = MySQL.MySQL('172.16.0.20')
    sql = "select family from tagsystem.Image where status=1 group by family"
    label = []
    i = 0
    for row in mysql_20.execute_query(sql):
        family = row[0]
        label.append((str(i), family))
        i += 1

    label = dict(label)
    return label


if __name__ == '__main__':
    print(index())