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

  public influential_features: string[] = ["email_address_risk_30day_window", "card_nb_tx_30day_window", "email_address_risk_7day_window", "diff_tx_time_in_hours"];
  public verbose_explanation: string[] = [];

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
        data: [0,0,0,0,0]
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

  createSentences(): void {
    let explanation: string = "";
    const sentence = this.influential_features[0];

    if (this.risk_score > 0.5) {
      const token0 = sentence.split("_")[0]
      if (token0 == "card"){
        const type = sentence.split("_")[1]
        if (type == "avg"){
          const number_days = "7";
          explanation = "The amount spent with this CARD during the last N days "
        }else if (type == "nb"){
          explanation = "The number of transactions made with this CARD during the last N days it's been higher than usual"
        }
      } else if (token0 == "email"){
        const type = sentence.split("_")[2]
        if (type == "risk"){
          explanation = "There have been previous transactions made with this EMAIL during the last N days that have been fraudulent."
        }else if (type == "nb"){
          explanation = "The number of transactions made with this EMAIL during the last N days it's been higher than usual"
        }
      } else if (token0 == "ip"){
        const type = sentence.split("_")[2]
        if (type == "risk"){
          explanation = "There have been previous transactions made with this IP during the last N days that have been fraudulent."
        }else if (type == "nb"){
          explanation = "The number of transactions made with this IP during the last N days it's been higher than usual"
        }
      }
    }
  }


}
