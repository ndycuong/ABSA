const express = require('express');
const cors = require('cors');
const axios = require('axios');
const bodyParser = require('body-parser');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');

const app = express();
const port = 5000;

app.use(cors());
app.use(bodyParser.json({ limit: '100mb' })); 
app.use(bodyParser.urlencoded({ limit: '100mb', extended: true })); 
app.use(express.static(path.join(__dirname, '../react_frontend/build')));

// Connect to the SQLite database
// "E:\sqlite-tools-win-x64-3460000\data1.db"
const db = new sqlite3.Database('E:/Downloads/webjs - Copy/python_server/model/data.db', (err) => {
    if (err) {
        console.error('Error opening database:', err);
    } else {
        console.log('Connected to the SQLite database.');
    }
});

app.post('/predict/absa1', async (req, res) => {
    try {
        const response = await axios.post('http://127.0.0.1:5001/predict/absa1', {
            text: req.body.text
        });
        res.json(response.data);
    } catch (error) {
        console.error("Error connecting to Flask server:", error);
        res.status(500).json({ error: "Failed to connect to Flask server" });
    }
});

app.post('/predict/absa2', async (req, res) => {
    try {
        const response = await axios.post('http://127.0.0.1:5001/predict/absa2', {
            text: req.body.text
        });
        res.json(response.data);
    } catch (error) {
        console.error("Error connecting to Flask server:", error);
        res.status(500).json({ error: "Failed to connect to Flask server" });
    }
});

app.post('/predict/absa3', async (req, res) => {
    try {
        const response = await axios.post('http://127.0.0.1:5001/predict/absa3', {
            text: req.body.text
        });
        res.json(response.data);
    } catch (error) {
        console.error("Error connecting to Flask server:", error);
        res.status(500).json({ error: "Failed to connect to Flask server" });
    }
});

app.post('/predict/sa', async (req, res) => {
    try {
        const response = await axios.post('http://127.0.0.1:5001/predict/sa', {
            text: req.body.text
        });
        res.json(response.data);
    } catch (error) {
        console.error("Error connecting to Flask server:", error);
        res.status(500).json({ error: "Failed to connect to Flask server" });
    }
});

app.post('/process_comments', async (req, res) => {
    try {
        console.log('Filtered comments to be sent:', req.body.comments); 

        const response = await axios.post('http://127.0.0.1:5001/process_comments', {
            comments: req.body.comments
        });
        res.json(response.data);
    } catch (error) {
        console.error("Error processing comments:", error);
        res.status(500).json({ error: "Failed to process comments" });
    }
});

app.get('/products', (req, res) => {
    db.all(`SELECT * FROM products`, [], (err, rows) => {
        if (err) {
            console.error(err);
            res.status(500).send("Internal Server Error");
        } else {
            res.json(rows);
        }
    });
});

// Fetch comments and aspects for a given product
app.get('/products/:productId/comments', (req, res) => {
    const { productId } = req.params;
    db.all(
      `SELECT comments.*, absa.aspect, absa.sentiment
       FROM comments
       LEFT JOIN absa ON comments.id = absa.comment_id
       WHERE comments.product_id = ?`,
      [productId],
      (err, rows) => {
        if (err) {
          console.error(err);
          res.status(500).json({ error: 'Failed to fetch comments' });
        } else {
          // Group comments by cmtid and aggregate aspects
          const commentsMap = {};
          rows.forEach(row => {
            if (!commentsMap[row.cmtid]) {
              commentsMap[row.cmtid] = {
                ...row,
                absa: []
              };
            }
            if (row.aspect && row.sentiment) {
              commentsMap[row.cmtid].absa.push([row.aspect, row.sentiment]);
            }
          });
          res.json(Object.values(commentsMap));
        }
      }
    );
  });


app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
