require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

// Configuración de conexión
const dbConfig = {
    user: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASS || 'postgres',
    host: process.env.DB_HOST || 'localhost',
    port: process.env.DB_PORT || 5432,
    database: process.env.DB_NAME || 'real_cordova_db'
};

const client = new Client(dbConfig);

async function initDB() {
    try {
        console.log("⏳ Conectando a PostgreSQL...");
        await client.connect();
        console.log("✅ Conexión a PostgreSQL establecida exitosamente desde Node.js (init.js).");
        
        // Crear tabla si no existe (inicialización de DB) alineado con schema.sql
        const createTableQuery = `
            CREATE TABLE IF NOT EXISTS llm_audit_logs (
                log_id SERIAL PRIMARY KEY,
                pregunta_usuario TEXT NOT NULL,
                sql_generado TEXT NOT NULL,
                sql_esperado TEXT,
                execution_accuracy NUMERIC(5,4) NOT NULL DEFAULT 0.0000,
                f1_score NUMERIC(5,4) NOT NULL DEFAULT 0.0000,
                latencia_ms INTEGER NOT NULL,
                tokens_prompt INTEGER,
                tokens_completion INTEGER,
                modelo VARCHAR(50) DEFAULT 'llama3.1:8b',
                error_msg TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        `;
        await client.query(createTableQuery);
        console.log("✅ Tabla 'llm_audit_logs' verificada/creada.");

        // Sincronizar archivo JSONL a DB
        const logFilePath = path.join(__dirname, process.env.LOG_DIR || 'logs', 'inferencias.jsonl');
        if (fs.existsSync(logFilePath)) {
            console.log(`⏳ Sincronizando logs desde: ${logFilePath}`);
            const fileStream = fs.createReadStream(logFilePath);
            const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

            for await (const line of rl) {
                if (!line.trim()) continue;
                try {
                    const record = JSON.parse(line);
                    const insertQuery = `
                        INSERT INTO llm_audit_logs 
                        (pregunta_usuario, sql_generado, sql_esperado, execution_accuracy, f1_score, latencia_ms, error_msg, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    `;
                    const values = [
                        record.pregunta_nl,
                        record.sql_generado,
                        record.sql_esperado || null,
                        record.ex_score || 0.0,
                        record.f1_score || 0.0,
                        Math.round(record.latencia_ms || 0),
                        record.error_msg || null,
                        record.timestamp || new Date()
                    ];
                    await client.query(insertQuery, values);
                } catch (parseErr) {
                    console.error("⚠️ Error parseando/insertando línea:", parseErr);
                }
            }
            console.log("✅ Sincronización de JSONL a PostgreSQL completada.");
            
            // Opcional: limpiar el archivo tras sincronizar
            // fs.truncateSync(logFilePath, 0);
        } else {
            console.log("ℹ️ No se encontró archivo de inferencias.jsonl para sincronizar.");
        }

    } catch (err) {
        console.error("❌ Error de Base de Datos en init.js:", err);
    } finally {
        await client.end();
        console.log("🔌 Conexión cerrada.");
    }
}

initDB();
