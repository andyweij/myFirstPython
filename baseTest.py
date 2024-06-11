from datetime import datetime

def create_exp_timeStamp(year):
    last_day_of_year = datetime(year, 12, 31, 23, 59, 59, 0)
    timestamp = last_day_of_year.timestamp()

    return timestamp


if __name__ == '__main__':
    timestamp=create_exp_timeStamp(2025)
    print(timestamp)
