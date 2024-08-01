import React, { useState, useEffect, useRef } from 'react';
import { Bar, Line, Pie } from 'react-chartjs-2';
import { Chart as ChartJS, registerables } from 'chart.js';
import axios from 'axios';
import 'chartjs-adapter-date-fns';
import './App.css'; // Make sure to import the CSS for styling
import MatrixChart from './MatrixChart'; // Import the custom Matrix chart component

// Register the necessary Chart.js components
ChartJS.register(...registerables);

const Dashboard = () => {
  const [products, setProducts] = useState([]);
  const [filteredData, setFilteredData] = useState([]);
  const [productType, setProductType] = useState('');
  const [productName, setProductName] = useState('');
  const [chartData, setChartData] = useState(null);
  const [tableData, setTableData] = useState([]);
  const [sentimentOverTime, setSentimentOverTime] = useState(null);
  const [pieChartData, setPieChartData] = useState(null);
  const [overallRating, setOverallRating] = useState(null);
  const [aspectSentimentOverTime, setAspectSentimentOverTime] = useState(null);
  const [totalComments, setTotalComments] = useState(0);
  const [randomComment, setRandomComment] = useState(null);
  const [analyzeClicked, setAnalyzeClicked] = useState(false);
  const sentimentScores = {
    positive: 1,
    neutral: 0,
    negative: -1,
  };

  const chartRef = useRef(null);

  useEffect(() => {
    // Fetch products from the API
    const fetchProducts = async () => {
      try {
        const response = await axios.get('http://localhost:5000/products'); // Adjust this URL as needed
        console.log('Fetched Products:', response.data); // Debug log
        setProducts(response.data);
      } catch (error) {
        console.error('Error fetching products:', error);
      }
    };
    fetchProducts();
  }, []);

  useEffect(() => {
    if (filteredData && filteredData.length > 0) {
      // Fetch comments for the filtered products
      const fetchCommentsForFilteredData = async () => {
        try {
          const productIds = filteredData.map((product) => product.id);
          const commentPromises = productIds.map((productId) =>
            axios.get(`http://localhost:5000/products/${productId}/comments`)
          );
          const commentResponses = await Promise.all(commentPromises);
          const commentsData = commentResponses.map((response) => response.data);
          console.log('Fetched Comments:', commentsData); // Debug log

          // Process the commentsData as before
          const totalCommentsCount = commentsData.reduce((acc, cur) => acc + cur.length, 0);
          setTotalComments(totalCommentsCount);

          const aspectCounts = commentsData.reduce((acc, cur) => {
            cur.forEach((comment) => {
              if (comment.absa) {
                comment.absa.forEach(([aspect, sentiment]) => {
                  if (!acc[aspect]) acc[aspect] = { positive: 0, neutral: 0, negative: 0 };
                  acc[aspect][sentiment.toLowerCase()] += 1;
                });
              }
            });
            return acc;
          }, {});

          console.log('Aspect Counts:', aspectCounts); // Debug log

          const labels = Object.keys(aspectCounts);
          const positiveData = labels.map((label) => aspectCounts[label].positive);
          const neutralData = labels.map((label) => aspectCounts[label].neutral);
          const negativeData = labels.map((label) => aspectCounts[label].negative);

          setChartData({
            labels,
            datasets: [
              { label: 'Positive', data: positiveData, backgroundColor: '#609966' },
              { label: 'Neutral', data: neutralData, backgroundColor: '#D3E4CD' },
              { label: 'Negative', data: negativeData, backgroundColor: '#EA5455' },
            ],
          });

          const tableRows = labels.map((label) => ({
            aspect: label,
            positive: aspectCounts[label].positive,
            neutral: aspectCounts[label].neutral,
            negative: aspectCounts[label].negative,
          }));

          setTableData(tableRows);

          // Calculate sentiment over time
          const sentimentTimeData = commentsData.reduce((acc, comments) => {
            comments.forEach((comment) => {
              const date = new Date(comment.ctime).toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
              if (!acc[date]) acc[date] = { positive: 0, neutral: 0, negative: 0 };
              if (comment.absa) {
                comment.absa.forEach(([aspect, sentiment]) => {
                  acc[date][sentiment.toLowerCase()] += 1;
                });
              }
            });
            return acc;
          }, {});

          console.log('Sentiment Time Data:', sentimentTimeData); // Debug log

          const sentimentTimeLabels = Object.keys(sentimentTimeData).sort((a, b) => new Date(a) - new Date(b));
          const sentimentCounts = sentimentTimeLabels.map((label) => sentimentTimeData[label]);

          const positiveSentimentData = sentimentCounts.map((counts) => counts.positive);
          const neutralSentimentData = sentimentCounts.map((counts) => counts.neutral);
          const negativeSentimentData = sentimentCounts.map((counts) => counts.negative);

          setSentimentOverTime({
            labels: sentimentTimeLabels,
            datasets: [
              {
                label: 'Positive',
                data: positiveSentimentData,
                backgroundColor: '#609966',
                borderColor: 'green',
                fill: false,
              },
              {
                label: 'Neutral',
                data: neutralSentimentData,
                backgroundColor: '#D3E4CD',
                borderColor: 'gray',
                fill: false,
              },
              {
                label: 'Negative',
                data: negativeSentimentData,
                backgroundColor: '#EA5455',
                borderColor: 'red',
                fill: false,
              },
            ],
          });

          // Calculate aspect sentiment over time
          const aspectSentimentTimeData = commentsData.reduce((acc, comments) => {
            comments.forEach((comment) => {
              const date = new Date(comment.ctime).toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
              if (!acc[date]) acc[date] = {};

              if (comment.absa) {
                comment.absa.forEach(([aspect, sentiment]) => {
                  if (!acc[date][aspect]) acc[date][aspect] = { positive: 0, neutral: 0, negative: 0 };
                  acc[date][aspect][sentiment.toLowerCase()] += 1;
                });
              }
            });
            return acc;
          }, {});

          console.log('Aspect Sentiment Time Data:', aspectSentimentTimeData); // Debug log

          const aspectSentimentDatasets = labels.flatMap((aspect) => [
            {
              label: `${aspect} Positive`,
              data: sentimentTimeLabels.map((date) => aspectSentimentTimeData[date][aspect]?.positive || 0),
              backgroundColor: 'rgba(0, 128, 0, 0.5)',
              stack: aspect,
            },
            {
              label: `${aspect} Negative`,
              data: sentimentTimeLabels.map((date) => aspectSentimentTimeData[date][aspect]?.negative || 0),
              backgroundColor: 'rgba(255, 0, 0, 0.5)',
              stack: aspect,
            },
          ]);

          setAspectSentimentOverTime({
            labels: sentimentTimeLabels,
            datasets: aspectSentimentDatasets,
          });

          // Calculate proportions for pie chart
          const totalPositive = positiveData.reduce((acc, val) => acc + val, 0);
          const totalNeutral = neutralData.reduce((acc, val) => acc + val, 0);
          const totalNegative = negativeData.reduce((acc, val) => acc + val, 0);
          const totalSentiments = totalPositive + totalNeutral + totalNegative;

          setPieChartData({
            labels: ['Positive', 'Neutral', 'Negative'],
            datasets: [
              {
                data: [
                  ((totalPositive / totalSentiments) * 100).toFixed(2),
                  ((totalNeutral / totalSentiments) * 100).toFixed(2),
                  ((totalNegative / totalSentiments) * 100).toFixed(2),
                ],
                backgroundColor: ['#609966', '#D3E4CD', '#EA5455'],
              },
            ],
          });

          const totalSentimentScore = commentsData.reduce((acc, cur) => {
            cur.forEach((comment) => {
              const sentimentScore = sentimentScores[comment.sa.toLowerCase()] || 0;
              acc += sentimentScore;
            });
            return acc;
          }, 0);

          const averageSentimentScore = (totalSentimentScore / totalCommentsCount).toFixed(2);
          setOverallRating(averageSentimentScore);

          // Select a random comment
          const allComments = commentsData.flat();
          const randomIndex = Math.floor(Math.random() * allComments.length);
          setRandomComment(allComments[randomIndex]);

        } catch (error) {
          console.error('Error fetching comments:', error);
        }
      };

      fetchCommentsForFilteredData();
    } else {
      setChartData(null);
      setTableData([]);
      setSentimentOverTime(null);
      setPieChartData(null);
      setOverallRating(null);
      setAspectSentimentOverTime(null);
      setTotalComments(0);
    }
  }, [filteredData]);

  const handleProductTypeChange = (e) => {
    setProductType(e.target.value);
    setProductName('');
  };

  const handleProductNameChange = (e) => {
    setProductName(e.target.value);
  };

  const handleAnalyzeClick = () => {
    if (products.length > 0) {
      setAnalyzeClicked(true);
      const filtered = products.filter((item) => item.product_type === productType && item.name === productName);
      console.log('Filtered Data:', filtered); // Debug log
      setFilteredData(filtered);
    }
  };

  const productTypes = products.length > 0 ? Array.from(new Set(products.map((item) => item.product_type))) : [];
  const productNames = products.length > 0 ? products.filter((item) => item.product_type === productType).map((item) => item.name) : [];
  const getProductLink = (product) => {
    if (product.platform === 'Shopee') {
      return `https://shopee.vn/product/${product.shopid}/${product.itemid}`;
    } else if (product.platform === 'Tiki') {
      return `https://tiki.vn/product/${product.itemid}`;
    }
    return '#';
  };

  const uniqueProductLinks = Array.from(new Set(filteredData.map((product) => product.id)))
    .map(id => filteredData.find(product => product.id === id));

  return (
    <div className="dashboard-container">
      <div className="controls">
        <label>
          Product Type:
          <select value={productType} onChange={handleProductTypeChange}>
            <option value="">Select Product Type</option>
            {productTypes.map((type, index) => (
              <option key={index} value={type}>{type}</option>
            ))}
          </select>
        </label>
        <label>
          Product Name:
          <select value={productName} onChange={handleProductNameChange}>
            <option value="">Select Product Name</option>
            {productNames.map((name, index) => (
              <option key={index} value={name}>{name}</option>
            ))}
          </select>
        </label>
      </div>
      <div className="analyze-button-container">
        <button onClick={handleAnalyzeClick}>Analyze</button>
      </div>

      <div className="dashboard-content">
        {analyzeClicked && (
          <div className="product-links">
            <h3>Link</h3>
            <ul>
              {uniqueProductLinks.map((product) => (
                <li key={product.id}>
                  <a href={getProductLink(product)} target="_blank" rel="noopener noreferrer">
                    {product.name} ({product.platform})
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
        {totalComments > 0 && (
          <div>
            <h3>Total Comments: {totalComments}</h3>
          </div>
        )}
        {/* {randomComment && (
          <div className="random-comment">
            <h3>Random Comment:</h3>
            <p><strong>Comment:</strong> {randomComment.comment}</p>
            <p><strong>Star:</strong> {randomComment.rating_star}</p>
            <p><strong>Sentiment:</strong> {randomComment.sa}</p>
            <p><strong>Aspect-based Sentiment:</strong></p>
            <ul>
              {randomComment.absa && randomComment.absa.filter(([aspect, sentiment]) => sentiment !== 'None').length > 0 ? (
                randomComment.absa
                  .filter(([aspect, sentiment]) => sentiment !== 'None')
                  .map(([aspect, sentiment], index) => (
                    <li key={index}>{aspect}: {sentiment}</li>
                  ))
              ) : (
                <li>No aspect detected</li>
              )}
            </ul>
          </div>
        )} */}
        {filteredData.length > 0 && (
          <div className="legend">
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <div style={{ backgroundColor: '#609966', width: '20px', height: '20px', marginRight: '5px' }}></div>
              <span>Positive</span>
              <div style={{ backgroundColor: '#D3E4CD', width: '20px', height: '20px', margin: '0 5px' }}></div>
              <span>Neutral</span>
              <div style={{ backgroundColor: '#EA5455', width: '20px', height: '20px', marginLeft: '5px' }}></div>
              <span>Negative</span>
            </div>
          </div>
        )}
        <div className="charts">
          {pieChartData && (
            <div className="chart-container">
              <Pie
                data={pieChartData}
                options={{
                  plugins: {
                    datalabels: {
                      formatter: (value, context) => {
                        const total = context.chart.data.datasets[0].data.reduce((acc, val) => acc + parseFloat(val), 0);
                        const percentage = ((value / total) * 100).toFixed(2) + '%';
                        return percentage;
                      },
                      color: '#fff',
                      font: {
                        weight: 'bold',
                        size: 14,
                      },
                    },
                    title: {
                      display: true,
                      text: 'Sentiment Distribution (%)',
                    },
                    legend: { display: false },
                  },
                }}
                style={{ width: '100%', maxHeight: '395px' }} // Adjust size here
              />
            </div>
          )}
          {chartData && (
            <div className="chart-container">
              <Bar
                ref={chartRef}
                data={chartData}
                options={{
                  scales: {
                    x: { type: 'category' },
                    y: { beginAtZero: true },
                  },
                  plugins: {
                    scales: {
                      x: { type: 'category', title: { display: true, text: 'Aspects' } },
                      y: { beginAtZero: true, title: { display: true, text: 'Count' } },
                    },
                    title: {
                      display: true,
                      text: 'Aspect Sentiment Counts',
                    },
                    legend: { display: false },
                  },
                }}
                style={{ width: '95%', maxHeight: '500px' }} // Adjust size here
              />
            </div>
          )}
          {sentimentOverTime && (
            <div className="chart-container full-width-chart">
              <Line
                data={sentimentOverTime}
                options={{
                  scales: {
                    x: { type: 'category', title: { display: true, text: 'Time' } },
                    y: { beginAtZero: true, title: { display: true, text: 'Count' } },
                  },
                  plugins: {
                    title: {
                      display: true,
                      text: 'Sentiment Over Time',
                    },
                    legend: { display: false },
                  },
                }}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
