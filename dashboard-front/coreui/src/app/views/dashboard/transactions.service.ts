import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Transaction } from './transaction';


@Injectable({
  providedIn: 'root'
})
export class TransactionsService {

  constructor(
    private http: HttpClient) { }

  transactionsUrl = "http://127.0.0.1:8000/transactions?size=2"
  
  public getTransactions(): Observable<Response> {
    return this.http.get<Response>(this.transactionsUrl, {});
  }
}
