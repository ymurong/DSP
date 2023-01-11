import { Component, OnInit } from '@angular/core';
import { UntypedFormControl, UntypedFormGroup } from '@angular/forms';
import { HttpClient, HttpParams } from '@angular/common/http';
import { TransactionsService } from './transactions.service'


import { DashboardChartsData, IChartProps } from './dashboard-charts-data';
import { Transaction } from './transaction';
import { json } from 'stream/consumers';


interface IUser {
  name: string;
  state: string;
  registered: string;
  country: string;
  usage: number;
  period: string;
  payment: string;
  activity: string;
  avatar: string;
  status: string;
  color: string;
  riskscore: number;
}

interface ITransaction {
  merchant: string;
  card_schema: string;
  is_credit: boolean;
  eur_amount: Number;
  ip_country: string;
  issuing_country: string;
  device_type: string;
  ip_address: string;
  email_address: string;
  card_number: string;
  shopper_interaction: string;
  zip_code: string;
  card_bin: string;
  has_fraudulent_dispute: boolean;
  is_refused_by_adyen: boolean;
  created_at: Date;
  updated_at: Date;
  psp_reference: bigint;
}

@Component({
  templateUrl: 'dashboard.component.html',
  styleUrls: ['dashboard.component.scss']
})
export class DashboardComponent implements OnInit {
  constructor(private chartsData: DashboardChartsData, private transactionsService: TransactionsService) {
  }

  public users: IUser[] = [
    {
      name: 'Yiorgos Avraamu',
      state: 'New',
      registered: 'Jan 1, 2021',
      country: 'Us',
      usage: 50,
      period: 'Jun 11, 2021 - Jul 10, 2021',
      payment: 'Mastercard',
      activity: '10 sec ago',
      avatar: './assets/img/avatars/1.jpg',
      status: 'accepted',
      color: 'success',
      riskscore: 0.16
    },
    {
      name: 'Avram Tarasios',
      state: 'Recurring ',
      registered: 'Jan 1, 2021',
      country: 'Br',
      usage: 10,
      period: 'Jun 11, 2021 - Jul 10, 2021',
      payment: 'Visa',
      activity: '5 minutes ago',
      avatar: './assets/img/avatars/2.jpg',
      status: 'refused',
      color: 'info',
      riskscore: 0.65
    },
    {
      name: 'Quintin Ed',
      state: 'New',
      registered: 'Jan 1, 2021',
      country: 'In',
      usage: 74,
      period: 'Jun 11, 2021 - Jul 10, 2021',
      payment: 'Stripe',
      activity: '1 hour ago',
      avatar: './assets/img/avatars/3.jpg',
      status: 'refused',
      color: 'warning',
      riskscore: 0.88
    },
    {
      name: 'Enéas Kwadwo',
      state: 'Sleep',
      registered: 'Jan 1, 2021',
      country: 'Fr',
      usage: 98,
      period: 'Jun 11, 2021 - Jul 10, 2021',
      payment: 'Paypal',
      activity: 'Last month',
      avatar: './assets/img/avatars/4.jpg',
      status: 'accepted',
      color: 'danger',
      riskscore: 0.45
    },
    {
      name: 'Agapetus Tadeáš',
      state: 'New',
      registered: 'Jan 1, 2021',
      country: 'Es',
      usage: 22,
      period: 'Jun 11, 2021 - Jul 10, 2021',
      payment: 'ApplePay',
      activity: 'Last week',
      avatar: './assets/img/avatars/5.jpg',
      status: 'accepted',
      color: 'primary',
      riskscore: 0.38
    },
    {
      name: 'Friderik Dávid',
      state: 'New',
      registered: 'Jan 1, 2021',
      country: 'Pl',
      usage: 43,
      period: 'Jun 11, 2021 - Jul 10, 2021',
      payment: 'Amex',
      activity: 'Yesterday',
      avatar: './assets/img/avatars/6.jpg',
      status: 'accepted',
      color: 'dark',
      riskscore: 0.38
    }
  ];
  public mainChart: IChartProps = {};
  public chart: Array<IChartProps> = [];
  public trafficRadioGroup = new UntypedFormGroup({
    trafficRadio: new UntypedFormControl('Month')
  });
  public classification_sensitivity: Number = 0.5;
  public transactions: Transaction[] = [];
  public numTransactions: number = 0;
  public sizePage: number = 0;

  ngOnInit(): void {
    this.initCharts();
    this.initTransactionList();
  }

  initTransactionList(): void {
    this.transactionsService.getTransactions().subscribe(
      (transactionList: any) => {
        console.log(transactionList)
        this.initializeTransactions(transactionList["items"])
        this.numTransactions = transactionList["total"]
        this.sizePage = transactionList["size"]
      }
    );
    console.log(this.transactions);
  }

  initializeTransactions(jsonResponse: any[]): void {
    for (var transaction of jsonResponse){
      const aux = transaction as ITransaction;
      this.transactions.push(aux);
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
