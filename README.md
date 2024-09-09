# CodeBusters - Leonardo 2024 Challenge

Notebook della challenge Leonardo 2024 affrontata il 9 settembre 2024.

Risultato: secondo premio. Traduzione sportiva: primo posto mancato per lo 0.001%, quindi ufficialmente "abbiamo perso per un capello". Ufficiosamente, il capello aveva un buon modello di ensemble.

## Challenge

La challenge richiedeva di classificare esempi multimodali usando:

- un'immagine dell'oggetto;
- il nome dell'oggetto;
- una descrizione testuale;
- una label target a 4 classi.

Il notebook assume questa struttura dati locale:

```text
dataset/
  train.csv
  test.csv
  images/
    train/
    test/
```

Il dataset non e' incluso nel repository.

## Soluzione

La soluzione usa un classificatore multimodale in PyTorch:

- `ViTModel` (`google/vit-base-patch16-224-in21k`) per estrarre feature visuali;
- `GPT2Model` e `GPT2Tokenizer` per codificare nome oggetto + descrizione;
- normalizzazione delle feature visuali e testuali;
- concatenazione delle due rappresentazioni;
- piccolo classificatore fully connected per predire una delle 4 classi.

Il training usa `AdamW`, validazione 90/10 sul training set e salvataggio del miglior modello su validation accuracy. Il notebook genera infine `submission.csv` per l'invio alla leaderboard.

## Uso

Installa l'ambiente con dipendenze bloccate:

```bash
uv sync --dev
```

Poi apri `CodeBusters - Leonardo.ipynb` nel tuo ambiente notebook usando il kernel del progetto.

## Note

Questo repo e' volutamente semplice: contiene l'artefatto della challenge, le dipendenze bloccate e il contesto minimo per capire come e' stata risolta.
