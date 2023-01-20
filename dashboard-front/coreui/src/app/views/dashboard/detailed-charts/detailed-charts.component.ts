import { Component, OnInit } from '@angular/core';
import {Chart, ChartConfiguration, ChartItem, registerables} from 'node_modules/chart.js'
import { MetricsService } from '../metrics.service';
Chart.register(...registerables);


@Component({
  selector: 'app-detailed-charts',
  templateUrl: './detailed-charts.component.html',
  styleUrls: ['./detailed-charts.component.scss']
})
export class DetailedChartsComponent implements OnInit{

  constructor(private metricsService: MetricsService) {}

  public threshold: number = 0.5;
  public dataLoaded: Promise<boolean> = Promise.resolve(false);
  public loading: Promise<boolean> = Promise.resolve(true);

  public data_block_rate: number[] = [];
  public data_fraud_rate: number[] = [];
  public data_revenue: number[] = [];
  public data_chargeback: number[] = [];

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
        data: this.data_block_rate,
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
        data: this.data_fraud_rate,
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
        data: this.data_revenue,
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
        data: this.data_chargeback,
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
    this.metricsService.getMonthMetrics(this.threshold).subscribe(
      (monthlyAnalytics: any) => {
        this.fillData(monthlyAnalytics);
        this.initializeChart();
        this.dataLoaded = Promise.resolve(true);
        this.loading = Promise.resolve(false);
      }
    )
    console.log(this.data_block_rate);
  }

  fillData(monthlyAnalytics: any[]): void {
    this.data_block_rate = []; this.data_fraud_rate = []; this.data_revenue = []; this.data_chargeback = [];
    for (let monthDict of monthlyAnalytics){
      this.data_block_rate.push(this.if0Nan(monthDict["block_rate"]));
      this.data_fraud_rate.push(this.if0Nan(monthDict["fraud_rate"]));
      this.data_revenue.push(this.if0Nan(monthDict["total_revenue"]));
      this.data_chargeback.push(this.if0Nan(monthDict["chargeback_costs"]));
    }
    console.log(this.data_revenue);
  }

  if0Nan(num: number): number{
    if (num == 0) {
      return NaN;
    } else {
      return num;
    }
  }

  initializeChart() {
    this.chartLineData = {
      labels: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
      datasets: [
        {
          label: 'Block Rate (%)',
          yAxisID: 'A',
          backgroundColor: 'rgba(9, 232, 98, 0.2)',
          borderColor: 'rgba(9, 232, 98, 1)',
          pointBackgroundColor: 'rgba(9, 232, 98, 1)',
          pointBorderColor: '#fff',
          data: this.data_block_rate,
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
          data: this.data_fraud_rate,
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
          data: this.data_revenue,
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
          data: this.data_chargeback,
          stack: 'true',
          type: 'bar'
        }
      ]
    };
  }


}
