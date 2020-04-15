import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MainComponent } from './main.component';
import { StreamVideoComponent } from './stream-video/stream-video.component';
import { SettingsComponent } from './settings/settings.component';



@NgModule({
  declarations: [
    MainComponent,
    StreamVideoComponent,
    SettingsComponent],
  imports: [
    CommonModule
  ]
})
export class MainModule { }
