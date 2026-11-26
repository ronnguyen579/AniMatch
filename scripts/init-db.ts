import 'dotenv/config';
import {readFileSync} from 'node:fs';
import pg from 'pg';
const pool=new pg.Pool({connectionString:process.env.DATABASE_URL});
await pool.query(readFileSync('database/schema.sql','utf8'));
await pool.end();
console.log('PostgreSQL schema ready.');
