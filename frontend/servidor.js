import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const porta = 3000;
const app = express();
app.use(express.json());

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

app.use(express.static(path.join(__dirname, 'src')))

app.get("/main", (req, res) => {
    res.sendFile(__dirname, "src/main/index.html")
})
app.get("/home", (req, res) => {
    res.sendFile(__dirname, "src/avaliacao/index2.html")
})
app.get("/contato", (req, res) => {
    res.sendFile(__dirname, "src/contato/index3.html")
})

app.listen(porta, () => {
    console.log('O servidor está rodando na porta: http://localhost:' + porta);
})