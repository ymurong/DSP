import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Transaction } from './transaction';


@Injectable({
  providedIn: 'root'
})
export class TransactionsService {

  constructor(
    private http: HttpClient) { }

  transactionsUrl = "http://127.0.0.1:8000/transactions"
  headers = new HttpHeaders();
  params = new HttpParams().set("size", 8)
  
  public getTransactions(current_page: number): Observable<Response> {
    this.params = this.params.set("page", current_page);
    return this.http.get<Response>(this.transactionsUrl, {headers: this.headers, params: this.params});
  }

  public getTransactionsFiltered(reference: number, accepted: boolean, rejected: boolean, current_page: number, merchant: string) {
    this.params = new HttpParams().set("size", 8).set("page", current_page);
    if (accepted != rejected){
      if (accepted){
        this.params = this.params.set("has_fraudulent_dispute", false);
      } else {
        this.params = this.params.set("has_fraudulent_dispute", true);
      }
    }
    if (reference != null){
      this.params = this.params.set("psp_reference", reference.toString());
    }
    if (merchant != "all"){
      this.params = this.params.set("merchant", merchant);
    }
    return this.http.get(this.transactionsUrl, {headers: this.headers, params: this.params}); 
    
  }
}
