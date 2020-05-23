import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Routes, RouterModule } from '@angular/router';
import { MainComponent } from './main.component';
import { StreamVideoComponent } from './stream-video/stream-video.component';
import { SettingsComponent } from './settings/settings.component';

import { WebcamModule } from 'ngx-webcam';
import { FormsModule } from '@angular/forms';

import { ApiService } from '../api.service';


const routes: Routes = [
  {path: 'dhh-app', component: MainComponent}
]

@NgModule({
  declarations: [
    MainComponent,
    StreamVideoComponent,
    SettingsComponent],
  imports: [
    FormsModule,
    WebcamModule,
    CommonModule,
    RouterModule.forChild(routes)
  ],
  exports: [
    RouterModule
  ],
  providers: [
    ApiService
  ]
})
export class MainModule { }
