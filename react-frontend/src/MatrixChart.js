// MatrixChart.js
import React from 'react';
import { Chart } from 'react-chartjs-2';
import { MatrixController, MatrixElement } from 'chartjs-chart-matrix';
import { Chart as ChartJS, registerables } from 'chart.js';

ChartJS.register(...registerables, MatrixController, MatrixElement);

const MatrixChart = ({ data, options }) => {
  return <Chart type='matrix' data={data} options={options} />;
};

export default MatrixChart;
