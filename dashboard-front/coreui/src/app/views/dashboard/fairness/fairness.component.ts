import { Component, Input, OnInit, OnChanges, SimpleChanges } from '@angular/core';
import { Router } from '@angular/router';


@Component({
  selector: 'app-fairness',
  templateUrl: './fairness.component.html',
  styleUrls: ['./fairness.component.scss']
})
export class FairnessComponent implements OnInit, OnChanges {

  @Input() labels: string[] = [""];
  @Input() data_accuracy: number[] = [0];
  @Input() data_fn: number[] = [0]

  public chartsLoaded: Promise<boolean> = Promise.resolve(false);

  public data_fn_rate = {
    labels: this.labels,
    datasets: [
      {
        label: 'False Negative Rate',
        backgroundColor: '#627596',
        data: this.data_fn
      }
    ]
  };

  public data_balanced_accuracy = {
    labels: this.labels,
    datasets: [
      {
        label: 'Balanced Accuracy',
        backgroundColor: '#627596',
        data: this.data_accuracy
      }
    ]
  };

  public constructor(public router:Router){}
  
  ngOnInit(): void {
    this.data_balanced_accuracy.labels = this.labels;
    this.data_fn_rate.labels = this.labels;
    this.data_balanced_accuracy.datasets[0].data = this.data_accuracy;
    this.data_fn_rate.datasets[0].data = this.data_fn;
  }

  ngOnChanges(changes: SimpleChanges): void {
    this.ngOnInit();
    this.chartsLoaded = Promise.resolve(true);
    //this.reloadComponent(true);
    }

  reloadComponent(self:boolean,urlToNavigateTo ?:string){
    //skipLocationChange:true means dont update the url to / when navigating
   console.log("Current route I am on:",this.router.url);
   const url=self ? this.router.url :urlToNavigateTo;
   this.router.navigateByUrl('/',{skipLocationChange:true}).then(()=>{
     this.router.navigate([`/${url}`]).then(()=>{
       console.log(`After navigation I am on:${this.router.url}`)
     })
   })
 }



  

}
