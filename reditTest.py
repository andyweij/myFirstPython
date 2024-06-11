import redis

if __name__ == '__main__':
    # 創建 Redis 客戶端
    r = redis.Redis(host='localhost', port=6379, db=0)

    # lredis-server保持開啓狀態，如果在客戶端設定了密碼 添加password=密碼即可
    pool = redis.ConnectionPool(host='127.0.0.1', port=6379, db=0)
    try:
        r.ping()
        print("Redis 連接成功！")
    except redis.ConnectionError as e:
        print(f"無法連接到 Redis: {e}")
    r = redis.StrictRedis(connection_pool=pool)
    # 字符串
    r.set('test', 'aaa')
    print(r.get('test'))
    # 列表
    # 注意python、lrange兩個range的範圍
    x = 0
    for x in range(0, 11):
        r.lpush('list', x)
        x = x + 1
    print(r.lrange('list', '0', '10'))

    # 雜湊
    dict_hash = {'name': 'tang', 'password': 'tang_passwd'}
    r.hmset('hash_test', dict_hash)
    print(r.hgetall('hash_test'))

    # 集合
    r.sadd('set_test', 'aaa', 'bbb')
    r.sadd('set_test', 'ccc')
    r.sadd('set_test', 'ddd')
    print(r.smembers('set_test'))

    # 有序集
    r.zadd('zset_test', {'aaa': 1, 'bbb': 1})
    r.zadd('zset_test', {'ccc': 1})
    r.zadd('zset_test', {'ddd': 1})
    print(r.zrange('zset_test', 0, 10))