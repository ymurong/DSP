import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Transaction } from './transaction';
import { environment } from 'src/environments/environment';


@Injectable({
  providedIn: 'root'
})
export class TransactionsService {

  constructor(
    private http: HttpClient) { }

  transactionsUrl = `${environment.API}/transactions`
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
/*
  public getMonthlyTransactions(month: number, year: number): Observable<Response> {
    let month_to = month + 1
    let year_to = year
    if (month_to > 12) {
      month_to = 1;
      year_to = year + 1;
    }
    
    const date_from = year.toString() + "-" + month.toString().padStart(2, '0') + "-01T00:00:00";
    const date_to = year_to.toString() + "-" + month_to.toString().padStart(2, '0') + "-01T00:00:00";

    const allElementsParams = new HttpParams().set("created_at_from", date_from).set("created_at_from", date_to);
    return this.http.get<Response>(this.transactionsUrl, {headers: this.headers, params: allElementsParams});
  }*/

  getExplainabilityScore(reference: number): Observable<Response> {
    let params = new HttpParams().set("explainer_name", "random_forest_lime");
    let explainabilityURI = `${environment.API}/transactions/${reference}/explainability_score`
    return this.http.get<Response>(explainabilityURI, {headers: this.headers, params: params})
  }
}
