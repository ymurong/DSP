import { Component, OnInit } from '@angular/core';
import { UntypedFormControl, UntypedFormGroup } from '@angular/forms';
import { HttpClient, HttpParams } from '@angular/common/http';
import { TransactionsService } from './transactions.service'


import { DashboardChartsData, IChartProps } from './dashboard-charts-data';
import { Transaction } from './transaction';
import { json } from 'stream/consumers';
import { throwIfEmpty } from 'rxjs';



interface ITransaction {
  merchant: string;
  card_schema: string;
  is_credit: boolean;
  eur_amount: number;
  ip_country: string;
  issuing_country: string;
  device_type: string;
  ip_address: string;
  email_address: string;
  card_number: string;
  shopper_interaction: string;
  zip_code: string;
  card_bin: string;
  has_fraudulent_dispute: boolean;
  is_refused_by_adyen: boolean;
  status_verbose: string;
  created_at: Date;
  created_at_string: string;
  updated_at: Date;
  psp_reference: bigint;
  prediction: RiskScore;
}

interface RiskScore {
  predict_proba: number;
  created_at: Date;
  updated_at: Date;
}

@Component({
  templateUrl: 'dashboard.component.html',
  styleUrls: ['dashboard.component.scss']
})
export class DashboardComponent implements OnInit {
  constructor(private chartsData: DashboardChartsData, private transactionsService: TransactionsService) {
  }

  public mainChart: IChartProps = {};
  public chart: Array<IChartProps> = [];
  public trafficRadioGroup = new UntypedFormGroup({
    trafficRadio: new UntypedFormControl('Month')
  });
  public classification_sensitivity: Number = 0.5;
  public transactions: ITransaction[] = [];
  public numTransactions: number = 0;
  public sizePage: number = 0;

  public acceptedTransaction = false;
  public rejectedTransaction = false;
  public pspIReference!: number;
  public filterMerchant: string = "all"

  public current_page: number = 1;
  public last_page: number = 0;

  public allMerchants: string[] = ["Merchant A", "Merchant B", "Merchant C", "Merchant D", "Merchant E"];

  ngOnInit(): void {
    this.initCharts();
    this.initTransactionList();
  }

  initTransactionList(): void {
    this.transactionsService.getTransactions(this.current_page).subscribe(
      (transactionList: any) => {
        console.log(transactionList)
        this.transactions = [];
        this.initializeTransactions(transactionList["items"])
        this.numTransactions = transactionList["total"]
        this.sizePage = transactionList["size"]
        this.last_page = Math.floor(this.numTransactions/this.sizePage) + 1
      }
    );
    console.log(this.transactions);
  }

  initializeTransactions(jsonResponse: any[]): void {
    for (var transaction of jsonResponse){
      let aux = transaction as ITransaction;
      aux = this.cleanAttributes(aux);
      this.transactions.push(aux);
    }
  }

  cleanAttributes(transaction: ITransaction){
    if (transaction.has_fraudulent_dispute){
      transaction.status_verbose = "rejected";
    } else {
      transaction.status_verbose = "accepted";
    }

    transaction.created_at_string = transaction.created_at.toLocaleString().replace("T", " - ");
    return transaction;
  }

  initCharts(): void {
    this.mainChart = this.chartsData.mainChart;
  }

  //submitForms(): void {
  //  document.getElementById("statusForm")
  //}

  applyFilters(): void {
    alert(this.filterMerchant);
    this.transactionsService.getTransactionsFiltered(this.pspIReference, this.acceptedTransaction, this.rejectedTransaction, this.current_page, this.filterMerchant).subscribe(
      (transactionList: any) => {
        this.transactions = [];
        this.initializeTransactions(transactionList["items"])
        this.numTransactions = transactionList["total"]
        this.sizePage = transactionList["size"]
        this.last_page = Math.floor(this.numTransactions/this.sizePage) + 1;
      }
    );
    //alert([this.acceptedTransaction, this.rejectedTransaction, this.pspIReference]);
  }

  goToPage(page_number: number): void {
    const last_page = Math.floor(this.numTransactions/this.sizePage) + 1;
    if (page_number != -1){
      this.current_page = page_number;
    } else {
      this.current_page = last_page;
    }
    this.initTransactionList();
  }

  moveToPage(pages: number): void {
    const last_page = Math.floor(this.numTransactions/this.sizePage) + 1;
    this.current_page = this.current_page + pages;
    if (this.current_page < 1){
      this.current_page = 1;
    } else if (this.current_page > last_page){
      this.current_page = last_page;
    }
    this.initTransactionList();
  }

  update_pagination() {
    const last_page = Math.floor(this.numTransactions/this.sizePage) + 1;
    /*if (false //this.current_page >) {

    } else if (false//this.current_page) {
      
    }*/
  }

  setTrafficPeriod(value: string): void {
    this.trafficRadioGroup.setValue({ trafficRadio: value });
    this.chartsData.initMainChart(value);
    this.initCharts();
  }
}
