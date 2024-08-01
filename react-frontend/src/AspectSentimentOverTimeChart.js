// components/AspectSentimentChart.js

import React from 'react';
import { Bar } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';

const AspectSentimentChart = ({ data }) => {
  const chartData = transformDataForChart(data);

  const options = {
    scales: {
      x: {
        type: 'time',
        time: {
          unit: 'month',
        },
      },
      y: {
        beginAtZero: true,
      },
    },
    plugins: {
      legend: {
        position: 'bottom',
      },
    },
  };

  return (
    <div>
      <h2>Aspect Sentiment Over Time</h2>
      <Bar data={chartData} options={options} />
    </div>
  );
};

const transformDataForChart = (aspectSentimentTimeData) => {
  const labels = [];
  const datasets = [];

  Object.keys(aspectSentimentTimeData).forEach(aspect => {
    const data = [];
    Object.keys(aspectSentimentTimeData[aspect]).forEach(date => {
      if (!labels.includes(date)) labels.push(date);
      const sentiments = aspectSentimentTimeData[aspect][date];
      const totalSentiments = sentiments.positive + sentiments.neutral + sentiments.negative;
      data.push(totalSentiments);
    });

    datasets.push({
      label: aspect,
      data,
      backgroundColor: getRandomColor(),
    });
  });

  labels.sort((a, b) => new Date(a) - new Date(b));

  return {
    labels,
    datasets,
  };
};

const getRandomColor = () => {
  const letters = '0123456789ABCDEF';
  let color = '#';
  for (let i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 16)];
  }
  return color;
};

export default AspectSentimentChart;
