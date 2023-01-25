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

  public influential_features: string[] = ["email_address_risk_30day_window", "card_nb_tx_30day_window", "email_address_risk_7day_window", "card_avg_amount_7day_window"];
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
        this.createReadableExplanation()
      }
    )
  }

  fillExplanationScores(explanationScores: any) {
    const data = [explanationScores["general_evidences"], explanationScores["ip_risk"], explanationScores["risk_card_behaviour"], explanationScores["risk_card_amount"], explanationScores["email_risk"]]
    this.chartRadarData.datasets[0].data = data;
  }

  createReadableExplanation(){
    this.verbose_explanation = [];
    for (let sentence of this.influential_features){
      this.createOneSentence(sentence);
    }
  }

  createOneSentence(sentence: string): void {
    let explanation: string = "";
    if (this.risk_score > 0.5) {
      const token0 = sentence.split("_")[0]
      const N = sentence.split("_").slice(-2)[0].slice(0, -3).toString();    
      const window = sentence.split("_").slice(-1).toString()
      if (window == "window") {
        if (token0 == "card"){
          const type = sentence.split("_")[1]
          if (type == "avg"){
            explanation = "The amount spent with this CARD during the last " + N + " days.";
          }else if (type == "nb"){
            explanation = "The number of transactions made with this CARD during the last "+ N +" days it's been higher than usual.";
          }
        } else if (token0 == "email"){
          const type = sentence.split("_")[2]
          if (type == "risk"){
            explanation = "There have been transactions made with this EMAIL during the last "+ N +" days that have been fraudulent.";
          }else if (type == "nb"){
            explanation = "The number of transactions made with this EMAIL during the last "+ N +" days it's been higher than usual.";
          }
        } else if (token0 == "ip"){
          const type = sentence.split("_")[2]
          if (type == "risk"){
            explanation = "There have been transactions made with this IP during the last "+ N +" days that have been fraudulent.";
          }else if (type == "nb"){
            explanation = "The number of transactions made with this IP during the last "+ N +" days it's been higher than usual.";
          }
        }
      } else if (sentence == "diff_tx_time_in_hours" || sentence == "is_night" || sentence == "is_weekend"){
        explanation = "The time in the day this transaction has been made differs from what is usual."
      } else if (sentence == "same_country"  || sentence == "is_diff_previous_ip_country"){
        explanation = "This transaction has been made in another country, which is unusual for this user."
      }
    }
    this.verbose_explanation.push(explanation);
  }


}
