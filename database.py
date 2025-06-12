import psycopg2
import pandas as pd
from configparser import ConfigParser

def config(filename='database.ini', section='postgresql'):
    """Parse database configuration from file"""
    parser = ConfigParser()
    parser.read(filename)
    
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(f'Section {section} not found in {filename}')
    
    return db

def connect():
    """Connect to the PostgreSQL database server"""
    conn = None
    try:
        # Read connection parameters
        params = config()
        
        # Connect to the PostgreSQL server
        conn = psycopg2.connect(**params)
        
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        if conn is not None:
            conn.close()
        return None

def query_to_dataframe(query):
    """Execute query and return results as DataFrame"""
    conn = connect()
    if conn:
        try:
            df = pd.read_sql_query(query, conn)
            return df
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            return None
        finally:
            conn.close()
    return None

def save_dataframe(df, table_name, if_exists='replace'):
    """Save DataFrame to database table"""
    conn = connect()
    if conn:
        try:
            # Create SQLAlchemy engine using the connection
            from sqlalchemy import create_engine
            from sqlalchemy.engine import URL
            
            params = config()
            url = URL.create(
                drivername='postgresql',
                username=params['user'],
                password=params['password'],
                host=params['host'],
                port=params['port'],
                database=params['database']
            )
            
            engine = create_engine(url)
            
            # Save DataFrame to database
            df.to_sql(table_name, engine, if_exists=if_exists, index=False)
            return True
        except Exception as error:
            print(error)
            return False
        finally:
            conn.close()
    return False