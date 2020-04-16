import { Component, OnInit } from '@angular/core';
import { ApiService } from '../../api.service';
import { DomSanitizer } from '@angular/platform-browser';

@Component({
  selector: 'app-settings',
  templateUrl: './settings.component.html',
  styleUrls: ['./settings.component.css']
})
export class SettingsComponent implements OnInit {

  image = null;

  constructor(
    private apiService: ApiService,
    private sanitizer: DomSanitizer
    ) { }

  ngOnInit(): void { }

  startStreaming() {
    this.apiService.startVideoStreaming().subscribe(
      data => {
        this.image = data;
        let objectURL = 'data:image/png;base64,' + data;
        this.image = this.sanitizer.bypassSecurityTrustUrl(objectURL);
      },
      error => {
        console.log("errffffffffffffffffffffor: ", error)
      }
    )
  }

  stopStreaming() {
    this.apiService.stopVideoStreaming().subscribe(
      data => {
        console.log("data: ", data)
      },
      error => {
        console.log("error: ", error)
      }
    )
  }

}
