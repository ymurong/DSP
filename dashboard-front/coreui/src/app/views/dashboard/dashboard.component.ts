import { Component, OnInit } from '@angular/core';
import { UntypedFormControl, UntypedFormGroup } from '@angular/forms';
import { MetricsService } from './metrics.service'
import { DashboardChartsData, IChartProps } from './dashboard-charts-data';


interface ModelRates {
  f1: number;
  recall: number;
  precision: number;
  auc: number;
  block_rate: number;
  fraud_rate:number;
}

interface ModelCosts{
  merchant: string;
  total_revenue: number;
  chargeback_costs: number;
}

interface Metrics {
  block_rate: number;
  fraud_rate: number;
  total_revenue: number;
  chargeback_costs: number;
}


@Component({
  templateUrl: 'dashboard.component.html',
  styleUrls: ['dashboard.component.scss']
})
export class DashboardComponent implements OnInit {
  constructor(private chartsData: DashboardChartsData, private metricsService: MetricsService) {
  }

  public mainChart: IChartProps = {};
  public chart: Array<IChartProps> = [];
  public trafficRadioGroup = new UntypedFormGroup({
    trafficRadio: new UntypedFormControl('Month')
  });
  public classification_sensitivity: number = 0.5;
  public current_threshold: number = this.classification_sensitivity;

  public model_rates!: ModelRates;
  public model_costs!: ModelCosts;
  public current_metrics_widget!: Metrics;
  public original_metrics_widget!: Metrics;

  public dataLoaded: Promise<boolean> = Promise.resolve(false);
  public loading: Promise<boolean> = Promise.resolve(true);
  public firstTime = true;

  public limitChargebackRario: number = 0.03;
  public over3percent = false;

  ngOnInit(): void {
    this.initCharts();
    this.updateMetrics();
  }

  updateMetrics(): void {
    this.updateAccuracy();
    this.updateRevenue();
  }

  updateAccuracy(): void {
    this.metricsService.getAccuracyMetrics(this.current_threshold).subscribe(
      (rates: any) => {
        this.model_rates = rates as ModelRates;
      }
    )
  }

  updateRevenue(): void {
    this.metricsService.getStoreCosts(this.current_threshold).subscribe(
      (rates: any) => {
        this.aggregateDataStores(rates);
        this.createMetricWidget();
        this.checkInconsistencies();
        if (this.firstTime){
          this.createOriginalMetricWidget();
          this.dataLoaded = Promise.resolve(true);
          this.loading = Promise.resolve(false);
          this.firstTime = false;
        }
      }
    )
  }

  aggregateDataStores(storeMeta: any[]){
    this.model_costs = {
      merchant : 'all',
      total_revenue: 0,
      chargeback_costs: 0
    };

    for (var transaction of storeMeta){
      const transactionObj = transaction as ModelCosts;
      this.model_costs.total_revenue += transactionObj.total_revenue;
      this.model_costs.chargeback_costs += transactionObj.chargeback_costs;
    }
  }

  createMetricWidget() {
    this.current_metrics_widget = {
      block_rate : this.model_rates.block_rate,
      fraud_rate: this.model_rates.fraud_rate,
      total_revenue: this.model_costs.total_revenue,
      chargeback_costs: this.model_costs.chargeback_costs
    };
  }

  createOriginalMetricWidget() {
    this.original_metrics_widget = {
      block_rate : this.model_rates.block_rate,
      fraud_rate: this.model_rates.fraud_rate,
      total_revenue: this.model_costs.total_revenue,
      chargeback_costs: this.model_costs.chargeback_costs
    };
  }

  checkInconsistencies() {
    if (this.current_metrics_widget.chargeback_costs/this.current_metrics_widget.total_revenue > this.limitChargebackRario){
      this.over3percent = true
    } else {
      this.over3percent = false
    }

  }

  initCharts(): void {
    this.mainChart = this.chartsData.mainChart;
  }

  setTrafficPeriod(value: string): void {
    this.trafficRadioGroup.setValue({ trafficRadio: value });
    this.chartsData.initMainChart(value);
    this.initCharts();
  }
}
