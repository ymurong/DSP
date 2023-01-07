import { Component, OnInit } from '@angular/core';
import { Chart } from 'chart.js';

@Component({
  selector: 'app-explanation',
  templateUrl: './explanation.component.html',
  styleUrls: ['./explanation.component.scss']
})
export class ExplanationComponent{

  chartRadarData = {
    labels: ['IP Risk Score', 'EMail Risk Score', 'Amount Risk Score', 'XXXXX', 'YYYYY'],
    datasets: [
      {
        label: 'Risk Score',
        backgroundColor: 'rgba(60, 179, 113, 0.2)',
        borderColor: 'rgb(0,206,116)',
        pointBackgroundColor: 'rgba(0,206,116,1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgba(179,181,198,1)',
        tooltipLabelColor: 'rgba(179,181,198,1)',
        data: [.85, .15, .90, .40, .40]
      }
    ]
  };

  chartRadarOptions = {
    responsive : true,
    plugins : {
      legend: {
        display: false
      },
      scales: {
        r: {
            angleLines: {
                display: false
            },
            suggestedMin: 0,
            suggestedMax: 1
        }
    }
    }
  }
}
