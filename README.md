# Leonardo Challenge 2024

Notebook della challenge di multimodal learning del Sapienza Training Camp organizzato con Leonardo Company S.p.A.

## Dettagli

- Evento: Sapienza Training Camp - Leonardo Company S.p.A.
- Periodo: 4-6 settembre 2024
- Titolo: Multimodal Learning from Language and Vision Processing Neural Networks
- Team: CodeBusters
- Risultato: secondo posto

Traduzione sportiva del risultato: primo posto mancato per lo 0.001%, quindi ufficialmente "abbiamo perso per un capello". Ufficiosamente, il capello aveva un buon modello di ensemble.

## Challenge

La challenge richiedeva di costruire un modello di multimodal learning per classificare il contesto nascosto di esempi descritti da piu' modalita' informative:

- un'immagine dell'oggetto;
- il nome dell'oggetto;
- una descrizione testuale;
- una label target a 4 classi.

In pratica, dato un dataset multimodale, bisognava predire una tra quattro classi usando testo, immagini e informazioni sull'oggetto rappresentato. Il rischio principale era che il modello si appoggiasse troppo all'oggetto esplicito, invece di catturare il vero segnale del contesto nascosto.

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
- rappresentazioni dell'oggetto disponibili nel dataset;
- normalizzazione delle feature visuali e testuali;
- concatenazione delle due rappresentazioni;
- piccolo classificatore fully connected per predire una delle 4 classi.

Il training usa `AdamW`, validazione 90/10 sul training set e salvataggio del miglior modello su validation accuracy. Il notebook genera infine `submission.csv` per l'invio alla leaderboard.

Un punto interessante della soluzione era il ragionamento sulla ridondanza dell'oggetto: siccome l'oggetto era gia' rappresentato in modo esplicito, aveva senso provare a rimuovere o compensare quella informazione dagli embedding testuali e visivi, cosi' da spingere il modello verso il target reale della challenge, cioe' il contesto.

## Descrizione breve

Partecipazione al Sapienza Training Camp organizzato con Leonardo Company, sviluppando un modello di apprendimento multimodale per la classificazione di contesti a partire da rappresentazioni testuali e visive. Realizzata un'architettura in PyTorch che combinava embedding testuali da GPT-2 e feature visive da Vision Transformer tramite proiezioni, normalizzazione e feature fusion neurale. Secondo posto con il team CodeBusters.

## Uso

Installa l'ambiente con dipendenze bloccate:

```bash
uv sync --dev
```

Poi apri `Leonardo Challenge 2024.ipynb` nel tuo ambiente notebook usando il kernel del progetto.

## Note

Questo repo e' volutamente semplice: contiene l'artefatto della challenge, le dipendenze bloccate e il contesto minimo per capire come e' stata risolta.
