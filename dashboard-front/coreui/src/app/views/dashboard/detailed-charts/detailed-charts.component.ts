import { Component, OnInit } from '@angular/core';
import {Chart, ChartConfiguration, ChartItem, registerables} from 'node_modules/chart.js'
Chart.register(...registerables);
import { TransactionsService } from '../transactions.service';


@Component({
  selector: 'app-detailed-charts',
  templateUrl: './detailed-charts.component.html',
  styleUrls: ['./detailed-charts.component.scss']
})
export class DetailedChartsComponent implements OnInit{

  constructor(private transactionsService: TransactionsService) {}

  chartLineData = {
    labels: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    datasets: [
      {
        label: 'Block Rate (%)',
        yAxisID: 'A',
        backgroundColor: 'rgba(9, 232, 98, 0.2)',
        borderColor: 'rgba(9, 232, 98, 1)',
        pointBackgroundColor: 'rgba(9, 232, 98, 1)',
        pointBorderColor: '#fff',
        data: [0.055, 0.055, 0.054, 0.052, 0.041, 0.051, 0.065, 0.069, 0.066, 0.065, 0.069, 0.062],
        stack: 'true',
        type: 'line'
      },
      {
        label: 'Fraud Rate (%)',
        yAxisID: 'A',
        backgroundColor: 'rgba(255, 28, 17, 0.2)',
        borderColor: 'rgba(255, 28, 17, 1)',
        pointBackgroundColor: 'rgba(255, 28, 17, 1)',
        pointBorderColor: '#fff',
        data: [0.060, 0.059, 0.050, 0.055, 0.045, 0.051, 0.055, 0.057, 0.057, 0.055, 0.056, 0.053],
        stack: 'true',
        type: 'line'
      },
      {
        label: 'Total Revenue',
        yAxisID: 'B',
        backgroundColor: 'rgba(9, 99, 233, 0.5)',
        borderColor: 'rgba(9, 99, 233, 1)',
        pointBackgroundColor: 'rgba(9, 99, 233, 1)',
        pointBorderColor: '#fff',
        data: [20000, 22000, 23000, 18000, 20000, 22000, 21000, 20000, 23000, 22000, 21000, 24000],
        stack: 'true',
        type: 'bar'
      },
      {
        label: 'Chargeback costs',
        
        yAxisID: 'B',
        backgroundColor: 'rgba(255, 171, 0, 0.5)',
        borderColor: 'rgba(255, 171, 0, 1)',
        pointBackgroundColor: 'rgba(255, 171, 0, 1)',
        pointBorderColor: '#fff',
        data: [1600, 1650, 1700, 1750, 1600, 1650, 1300, 1250, 1250, 1300, 1350, 1400],
        stack: 'true',
        type: 'bar'
      }
    ]
  };

  vLine = {
    annotation: {
      annotations: {
        line1: {
          type: 'line',
          xMin: "May",
          xMax: "May",
          borderColor: 'rgb(255, 99, 132)',
          borderWidth: 2,
        }
      }
    }
  };  


  ngOnInit(): void {
    this.createChart()
  }

  createChart(): void {
    for (let i = 0; i < 12; i++) {
      this.transactionsService.getMonthlyTransactions(1, 2021).subscribe(
        (transactionList: any) => {
          console.log(transactionList);
        }
      )
    }
    alert("Hola");
  }

}
