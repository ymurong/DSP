import { Component, OnInit } from '@angular/core';
import { TransactionsService } from '../transactions.service'



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
  selector: 'app-historical-transactions',
  templateUrl: './historical-transactions.component.html',
  styleUrls: ['./historical-transactions.component.scss']
})
export class HistoricalTransactionsComponent implements OnInit {
  constructor(private transactionsService: TransactionsService) {}

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
    this.initTransactionList();
  }

  initTransactionList(): void {
    this.transactionsService.getTransactions(this.current_page).subscribe(
      (transactionList: any) => {
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
    if (transaction.prediction.predict_proba > this.classification_sensitivity){
      transaction.status_verbose = "rejected";
    } else {
      transaction.status_verbose = "accepted";
    }

    transaction.created_at_string = transaction.created_at.toLocaleString().replace("T", " - ");
    return transaction;
  }

  
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

}
