import { Component, OnInit, Input } from '@angular/core';
import { Chart } from 'chart.js';

@Component({
  selector: 'app-explanation',
  templateUrl: './explanation.component.html',
  styleUrls: ['./explanation.component.scss']
})
export class ExplanationComponent implements OnInit{

  @Input() psp_reference: bigint = BigInt(0);
  @Input() risk_score: number = 0;

  chartRadarData = {
    labels: ['Circumstancial Evidences Score', 'IP Score', 'Card Behaviour Score', 'Amount Spent Score', 'E-Mail Score'],
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
            suggestedMin: -2,
            suggestedMax: 1
        }
    }
    }
  }

  ngOnInit(): void {
    console.log(this.psp_reference);
  }
}
