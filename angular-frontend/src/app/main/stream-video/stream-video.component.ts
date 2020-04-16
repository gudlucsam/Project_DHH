import { Component, OnInit } from '@angular/core';
import { ApiService } from '../../api.service';

@Component({
  selector: 'app-stream-video',
  templateUrl: './stream-video.component.html',
  styleUrls: ['./stream-video.component.css']
})
export class StreamVideoComponent implements OnInit {

  infoApi = '';
  constructor(private apiService: ApiService) { }

  ngOnInit(): void {
    this.apiService.getApiInfo().subscribe(
      data => {
        console.log("data: ", data);
      },
      error => console.log("error: ", error)
    )
  }

}
