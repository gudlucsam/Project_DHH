import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';
import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { MainModule } from './main/main.module'
import { AppComponent } from './app.component';
import {NgbModule} from '@ng-bootstrap/ng-bootstrap';
import { HttpClientModule } from '@angular/common/http';
import { WebcamModule } from 'ngx-webcam'

const routes: Routes = [
  {path: '', pathMatch: 'full', redirectTo: 'dhh-app'}
]

@NgModule({
  declarations: [
    AppComponent
  ],
  imports: [
    WebcamModule,
    FormsModule,
    HttpClientModule,
    NgbModule,
    BrowserModule,
    MainModule,
    RouterModule.forRoot(routes)
  ],
  exports: [
    RouterModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
