async function analisarAvaliacao() {
    const avaliacao = document.getElementById("inputAvaliacao").value;

    const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ avaliacao })
    });

    const data = await response.json();
    document.getElementById("resultado").innerText = "Sentimento: " + data.sentimento;
}
