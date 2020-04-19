import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  baseUrl = 'http://127.0.0.1:5000/ml/api/v1.0/'
  headers = new HttpHeaders({
    'Content-Type': 'application/json'
  })

  constructor(private httpClient: HttpClient) { }

  getApiInfo(){
    return this.httpClient.get(`${this.baseUrl}info`, {headers: this.headers});
  }

  startVideoStreaming(){
    const headers = new HttpHeaders({
      'Content-Type': 'multipart/x-mixed-replace'
    })
    const body = JSON.stringify({'status': true});
    return this.httpClient.post(`${this.baseUrl}md/vf`, body, {headers: headers});
  }

  stopVideoStreaming(){
    const body = JSON.stringify({'status': false});
    return this.httpClient.post(`${this.baseUrl}md/vf`, body, {headers: this.headers});
  }

}
