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
  public accepted :boolean = this.risk_score < 0.5;

  public verbose_explanation: string[] = [];
  public general_explanations: string[] = [];

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
    this.createRadarChart();
    this.getMostInfluentialFeatures();
    this.accepted = this.risk_score < 0.5;
  }

  createRadarChart(): void {
    this.explanationService.getExplainabilityScore(Number(this.psp_reference)).subscribe(
      (explanationScores: any) => {
        this.fillExplanationScores(explanationScores)
      }
    )
  }

  getMostInfluentialFeatures(): void {
    this.explanationService.getInfluentialFeatures(Number(this.psp_reference)).subscribe(
      (influential_features: any[]) => {
        this.createReadableExplanation(influential_features)
      }
    )
  }

  fillExplanationScores(explanationScores: any) {
    const data = [explanationScores["general_evidences"], explanationScores["ip_risk"], explanationScores["risk_card_behaviour"], explanationScores["risk_card_amount"], explanationScores["email_risk"]]
    this.chartRadarData.datasets[0].data = data;
  }

  createReadableExplanation(influential_features: string[]){
    this.verbose_explanation = [];
    for (let sentence of influential_features){
      this.createOneSentence(sentence);
    }
    if (this.general_explanations.length > 0){
      let features = ""
      for (let explanation of this.general_explanations){
        features += explanation;
        features +=  " and "
      }
      let last_sentence = "General evidences such as (" + features.slice(0, -5) + ") have been decisive to block this transaction";
      this.verbose_explanation.push(last_sentence);
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
        explanation = "The time in the day this transaction has been made differs from what is usual.";
      } else if (sentence == "same_country"  || sentence == "is_diff_previous_ip_country"){
        explanation = "This transaction has been made in another country, which is unusual for this user.";
      } else if (sentence == "is_credit"){
        this.general_explanations.push("having used a credit card")
      } else if (sentence == "shopper_interaction_POS"){
        this.general_explanations.push("being an online transaction")
      }
    }
    if (explanation != ""){
      this.verbose_explanation.push(explanation);
    } 
  }


}
