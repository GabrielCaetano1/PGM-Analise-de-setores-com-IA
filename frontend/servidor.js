import express from 'express'
import dotenv from 'dotenv'

dotenv.config();
const app = express();
app.use(express.json());
app.use('./user')

const porta = process.env.PORTA

app.listen(porta, () => {
    console.log('O servidor está rodando na porta: http://localhost: ' + porta);
})