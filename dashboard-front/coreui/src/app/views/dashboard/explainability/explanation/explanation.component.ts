import { Component, OnInit, Input } from '@angular/core';
import { Chart } from 'chart.js';
import { TransactionsService } from '../../transactions.service';

@Component({
  selector: 'app-explanation',
  templateUrl: './explanation.component.html',
  styleUrls: ['./explanation.component.scss']
})
export class ExplanationComponent implements OnInit{
  constructor(
    private explanationService: TransactionsService){}

  @Input() psp_reference: bigint = BigInt(0);
  @Input() risk_score: number = 0;

  chartRadarData = {
    labels: ['General Evidences Score', 'IP Score', 'Card Behaviour Score', 'Amount Spent Score', 'E-Mail Score'],
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

  chartOptions = {
    elements: {
      line: {
        borderWidth: 3
      }
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
  };

  ngOnInit(): void {
    this.createRadarChart()
  }

  createRadarChart(): void {
    this.explanationService.getExplainabilityScore(Number(this.psp_reference)).subscribe(
      (explanationScores: any) => {
        this.fillExplanationScores(explanationScores)
      }
    )
  }

  fillExplanationScores(explanationScores: any) {
    const data = [explanationScores["general_evidences"], explanationScores["ip_risk"], explanationScores["risk_card_behaviour"], explanationScores["risk_card_amount"], explanationScores["email_risk"]]
    this.chartRadarData.datasets[0].data = data;
  }


}
