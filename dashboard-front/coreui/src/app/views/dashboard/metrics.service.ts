import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from 'src/environments/environment';


@Injectable({
  providedIn: 'root'
})
export class MetricsService {

  constructor(
    private http: HttpClient
  ) { }
  
  metadataMetrics = `${environment.API}/metadata/classifier/metrics/random_forest`
  metadataCosts = `${environment.API}/metadata/store/metrics`
  headers = new HttpHeaders();
  
  getAccuracyMetrics(threshold: number): Observable<Response> {
    let params = new HttpParams().set("threshold", threshold);
    return this.http.get<Response>(this.metadataMetrics, {headers: this.headers, params: params});
  }

  getStoreCosts(threshold: number): Observable<Response[]> {
    let params = new HttpParams().set("threshold", threshold);
    return this.http.get<Response[]>(this.metadataCosts, {headers: this.headers, params: params})
  }
}
