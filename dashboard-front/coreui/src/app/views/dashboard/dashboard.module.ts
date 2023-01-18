import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule,FormsModule } from '@angular/forms';
import { TooltipModule } from '@coreui/angular';
import { NgxChartsModule } from '@swimlane/ngx-charts';

import {
  AvatarModule,
  ButtonGroupModule,
  ButtonModule,
  CardModule,
  FormModule,
  GridModule,
  NavModule,
  ProgressModule,
  TableModule,
  TabsModule
} from '@coreui/angular';
import { IconModule } from '@coreui/icons-angular';
import { ChartjsModule } from '@coreui/angular-chartjs';
import { DashboardRoutingModule } from './dashboard-routing.module';
import { DashboardComponent } from './dashboard.component';
import { WidgetsModule } from '../widgets/widgets.module';
import { ExplanationComponent } from './explainability/explanation/explanation.component';
import { DetailedChartsComponent } from './detailed-charts/detailed-charts.component';
import { HttpClientModule } from '@angular/common/http';
import { HistoricalTransactionsComponent } from './historical-transactions/historical-transactions.component';

@NgModule({
  imports: [
    DashboardRoutingModule,
    CardModule,
    NavModule,
    IconModule,
    TabsModule,
    CommonModule,
    GridModule,
    ProgressModule,
    ReactiveFormsModule,
    ButtonModule,
    FormModule,
    FormsModule,
    ButtonModule,
    ButtonGroupModule,
    ChartjsModule,
    AvatarModule,
    TableModule,
    WidgetsModule,
    FormsModule,
    TooltipModule,
    NgxChartsModule,
    HttpClientModule
  ],
  declarations: [DashboardComponent, ExplanationComponent, DetailedChartsComponent, HistoricalTransactionsComponent]
})
export class DashboardModule {
}
